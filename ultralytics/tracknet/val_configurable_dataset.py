import json
from matplotlib import patches, pyplot as plt
import torch
from torch.utils.data import Dataset
import cv2
import hashlib
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from functools import lru_cache
from glob import glob

from ultralytics.tracknet.utils.preprocess import preprocess_csvV4
from ultralytics.tracknet.utils.preprocess import preprocess_csv, preprocess_csvV5

class TrackNetValConfigurableDataset(Dataset):
    """
    - 採用分離式設計：與訓練集完全不同的 match
    - 不進行 hit 樣本擴充
    - 用於評估模型的泛化能力
    """
    
    # Debug Setup
    DEBUG_BACKGROUND = False 

    DEBUG_SAMPLE_STRATEGY = 'per_match'
    DEBUG_PER_MATCH_COUNT = 3     
    DEBUG_INTERVAL = 100            
    DEBUG_BLOCK_SIZE = 20           
    DEBUG_BLOCK_INTERVAL = 1000     
    
    
    def __init__(self, root_dir, num_input=10, transform=None, prefix='',
                 background_method='mean'):
        """
        Args:
            background_method (str): 背景去除方法
                - 'none': 不使用背景去除
                - 'median': 使用中位數 (慢但穩健)
                - 'mean': 使用平均數 (快速)
        """

        print(f"\n========== VAL_CONFIGURABLE_DATASET LOADED ==========\nroot_dir: {root_dir}\n{'='*53}\n", flush=True)

        self.match_mog2 = {}
        self.root_dir = root_dir
        self.transform = transform
        self.num_input = num_input
        self.samples = []
        self.prefix = prefix
        
        # ============ Background removal setup ============
        self.background_method = background_method
        print(f"{'='*60}")
        print(f"[驗證集背景去除] Method: {self.background_method.upper()}")
        print(f"[驗證集背景去除] Debug: {self.DEBUG_BACKGROUND}")
        if self.DEBUG_BACKGROUND:
            strategy_desc = {
                'all': '保存所有樣本（慢，不推薦）',
                'per_match': f'每個match保存前{self.DEBUG_PER_MATCH_COUNT}個 ⭐',
                'interval': f'每{self.DEBUG_INTERVAL}個樣本保存1個',
                'block': f'每{self.DEBUG_BLOCK_INTERVAL}個開始保存連續{self.DEBUG_BLOCK_SIZE}個'
            }
            desc = strategy_desc.get(self.DEBUG_SAMPLE_STRATEGY, self.DEBUG_SAMPLE_STRATEGY)
            print(f"[驗證集背景去除] Debug抽樣: {desc}")
        print(f"{'='*60}\n")
        self.root_dir = root_dir
        self.transform = transform
        self.num_input = num_input
        self.samples = []
        self.prefix = prefix
        
        # 驗證集配置
        self.path_counts = {
            "sportxai_rally_test": 2000,           
            "sportxai_serve_machine_test": 1000,               
        }

        self.idx = set()

        # Traverse all matches
        last_len = 0
        for match_name in glob("*/", root_dir=root_dir):
            match_name = match_name.strip('/')

            match_dir_path = os.path.join(root_dir, match_name)

            # Check if it is a match directory
            if not os.path.isdir(match_dir_path):
                continue

            if match_name in self.path_counts:
                image_count = len(glob(os.path.join(self.root_dir, f"{match_name}/", "frame/", "*/", "*.png")))
                total_samples = image_count

                with tqdm(total=total_samples, desc=f"Processing {match_name}", miniters=1, smoothing=1) as pbar:
                    self.read_match(match_name, pbar)
            print(f"Total samples for {match_name}: {len(self.samples)-last_len}\n")
            last_len = len(self.samples)


    def read_match(self, match_name, pbar):
        # get metadata from metadata.json
        metadata_path = os.path.join(self.root_dir, match_name, 'metadata.json')
        # 讀取 JSON 檔案
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        head_width = data['calibration']['near_camera_head_width_px']

        video_dir = os.path.join(self.root_dir, match_name, 'video')
        csv_dir = os.path.join(self.root_dir, match_name, 'csv')

        pbar.set_description(f'{self.prefix} Generating image cache: {match_name}/ ')

        if match_name in self.path_counts:
            # gather both mp4 and avi files
            video_files = sorted(
                glob("*.mp4", root_dir=video_dir) + glob("*.avi", root_dir=video_dir)
            )

            samples_added_count = 0

            # Traverse all videos in the match directory
            for video_file in video_files:
                video_path = os.path.join(video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                cap.release()

                # base name without extension
                video_base, _ = os.path.splitext(video_file)

                csv_file = os.path.join(csv_dir, video_base + "_ball" + '.csv')
                if not os.path.isfile(csv_file):
                    continue

                ball_trajectory_df = self.__preprocess_csv(csv_file, fps, head_width)

                frame_dir = os.path.join(self.root_dir, match_name, 'frame', video_base)
                if not os.path.isdir(frame_dir):
                    continue

                img_files = sorted(glob("*.png", root_dir=frame_dir),
                                   key=lambda x: int(os.path.splitext(x)[0]))
                total_img_len = len(img_files)
                limit_count = self.path_counts[match_name]
                min_len = min(limit_count, total_img_len)
                if min_len == 0:
                    continue

                img = cv2.imread(os.path.join(frame_dir, img_files[0]))
                if img is None:
                    continue
                height, width, _ = img.shape

                # Create sliding windows of num_input frames
                # 驗證集：不進行 hit 樣本擴充，直接用原始樣本
                for i in range(min_len - (self.num_input-1)):
                    if samples_added_count >= limit_count:
                        break
                    frames = img_files[i: i + self.num_input]

                    target = ball_trajectory_df.iloc[i: i + self.num_input].values
                    target = self.transform_coordinates(target, width, height)

                    # Avoid invalid data
                    if len(frames) == self.num_input and len(target) == self.num_input:
                        npy_path = self.img_cache_dir(match_name, video_base, frames)

                        self.samples.append({
                            "match_name": match_name,
                            "video_name": video_base,
                            "cache_npy": npy_path,
                            "img_files": frames,
                            "target": target
                        })

                        self.img_cache(match_name, video_base, frames, npy_path)

                        pbar.update(1)

                        samples_added_count += 1
                # 驗證集只用 step=2 的下采樣，保持簡單
                """
                valid_steps = [2]

                for step in valid_steps:
                    num_frames_needed = self.num_input * step
                    max_start_idx = len(img_files) - num_frames_needed + 1

                    for i in range(max_start_idx):
                        frames = img_files[i: i + num_frames_needed: step]
                        target = ball_trajectory_df.iloc[i: i + num_frames_needed: step].values
                        target = self.transform_coordinates(target, width, height)

                        if len(frames) == self.num_input and len(target) == self.num_input:
                            npy_path = self.img_cache_dir(match_name, video_base, frames)

                            sample = {
                                "match_name": match_name,
                                "video_name": video_base,
                                "cache_npy": npy_path,
                                "img_files": frames,
                                "target": target
                            }

                            self.samples.append(sample)
                            self.img_cache(match_name, video_base, frames, npy_path)

                            pbar.update(1)
                """

    def img_cache_dir(self, match_name, video_name, img_files):
        """Generate cache directory and filename based on input images"""
        s = '|'.join([match_name]+[video_name]+img_files)
        filename = hashlib.sha1(s.encode('utf-8')).hexdigest()

        if filename in self.idx:
            raise Exception('DUP: '+filename)
        self.idx.add(filename)

        # 重要修正：Cache 路徑需要包含 background_method
        # 這樣不同背景方法會有獨立的 cache，避免混用
        # d = os.path.join(self.root_dir, ".cache", self.background_method, filename[:2], filename[2:4])

        cache_base = "/ssd2/tracknet_cache/train_data" if "train_data" in self.root_dir else "/ssd2/tracknet_cache/val_data"
        d = os.path.join(cache_base, self.background_method, filename[:2], filename[2:4])

        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, f"{filename}.npy")
        return f

    def img_cache(self, match_name, video_name, img_files, npy_path):
        """
        生成並快取影像
        與訓練集相同的處理邏輯
        """
        if os.path.isfile(npy_path):
            return

        # generate cache
        frames = [cv2.imread(os.path.join(self.root_dir, match_name, 'frame', video_name, fp), cv2.IMREAD_GRAYSCALE).astype(np.float32) 
                for fp in img_files]
        frames = np.array(frames)

        # ==================== Background Removal ====================
        if self.background_method == 'none':
            processed_frames = frames
            bg_frame = None
            
        elif self.background_method == 'median':
            # 中位數: 對雜訊穩健,但計算較慢 O(n log n)
            bg_frame = np.median(frames, axis=0).astype(np.float32)
            processed_frames = (frames - bg_frame).astype(np.float32)
            
        elif self.background_method == 'mean':
            # 平均數: 計算快速 O(n),推薦使用
            bg_frame = np.mean(frames, axis=0).astype(np.float32)
            processed_frames = (frames - bg_frame).astype(np.float32)
            
        else:
            raise ValueError(
                f"未知的 background_method: '{self.background_method}'\n"
                f"有效選項: 'none', 'median', 'mean'"
            )
        
        # Debug: 智能採樣保存圖片（減少輸出數量）
        if self.DEBUG_BACKGROUND and bg_frame is not None:
            # 從檔名提取 frame number (例如 "250.png" -> 250)
            try:
                frame_num = int(os.path.splitext(img_files[0])[0])
                should_save = False
                
                # 根據策略決定是否保存
                if self.DEBUG_SAMPLE_STRATEGY == 'all':
                    should_save = True
                    
                elif self.DEBUG_SAMPLE_STRATEGY == 'per_match':
                    if not hasattr(self, '_match_debug_counts'):
                        self._match_debug_counts = {}
                    if match_name not in self._match_debug_counts:
                        self._match_debug_counts[match_name] = 0
                    if self._match_debug_counts[match_name] < self.DEBUG_PER_MATCH_COUNT:
                        self._match_debug_counts[match_name] += 1
                        should_save = True
                
                elif self.DEBUG_SAMPLE_STRATEGY == 'interval':
                    if not hasattr(self, '_global_sample_count'):
                        self._global_sample_count = 0
                    self._global_sample_count += 1
                    if self._global_sample_count % self.DEBUG_INTERVAL == 0:
                        should_save = True
                
                elif self.DEBUG_SAMPLE_STRATEGY == 'block':
                    segment = frame_num // self.DEBUG_BLOCK_INTERVAL
                    offset_in_segment = frame_num % self.DEBUG_BLOCK_INTERVAL
                    if offset_in_segment < self.DEBUG_BLOCK_SIZE:
                        should_save = True
                
                if should_save:
                    self._save_debug_images(match_name, img_files[0], frames[0], bg_frame, processed_frames[0])
                    
            except (ValueError, IndexError):
                # 如果檔名格式不符預期，使用 per_match 策略
                if not hasattr(self, '_match_debug_counts'):
                    self._match_debug_counts = {}
                if match_name not in self._match_debug_counts:
                    self._match_debug_counts[match_name] = 0
                if self._match_debug_counts[match_name] < self.DEBUG_PER_MATCH_COUNT:
                    self._match_debug_counts[match_name] += 1
                    self._save_debug_images(match_name, img_files[0], frames[0], bg_frame, processed_frames[0])
        # ==================================================
        
        images = []
        for i, processed_frame in enumerate(processed_frames):
            img = self.pad_to_square(processed_frame)
            img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=0)
            images.append(img)
        img = np.concatenate(images, 0)

        np.save(npy_path, img)

    def _save_debug_images(self, match_name, img_file, original_frame, bg_frame, processed_frame):
        """
        保存 debug 圖片（驗證集版本）
        
        會同時保存：
        - 原始影格
        - 背景影像  
        - 去背結果（原始差值）
        - 去背結果（+128 視覺化）
        - 四合一對比圖
        """
        try:
            debug_dir = os.path.join(self.root_dir, '.cache', 'debug', match_name)
            os.makedirs(debug_dir, exist_ok=True)
            
            base_name = os.path.splitext(img_file)[0]
            
            # 1. 保存原始影格
            cv2.imwrite(
                os.path.join(debug_dir, f"original_{base_name}.png"),
                original_frame.astype(np.uint8)
            )
            
            # 2. 保存背景
            cv2.imwrite(
                os.path.join(debug_dir, f"background_{self.background_method}_{base_name}.png"),
                bg_frame.astype(np.uint8)
            )
            
            # 3. 保存去背結果 - 原始版本
            raw_result = np.clip(processed_frame, -128, 127)
            raw_visual = ((raw_result + 128)).astype(np.uint8)
            cv2.imwrite(
                os.path.join(debug_dir, f"processed_raw_{self.background_method}_{base_name}.png"),
                raw_visual
            )
            
            # 4. 保存去背結果 - 視覺化版本（+128）
            visual_result = np.clip(processed_frame + 128, 0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(debug_dir, f"processed_visual_{self.background_method}_{base_name}.png"),
                visual_result
            )
            
            # 5. 創建四合一對比圖
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # 左上: 原始影格
            axes[0, 0].imshow(original_frame, cmap='gray', vmin=0, vmax=255)
            axes[0, 0].set_title('原始影格', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 右上: 背景
            axes[0, 1].imshow(bg_frame, cmap='gray', vmin=0, vmax=255)
            axes[0, 1].set_title(f'背景 ({self.background_method})', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            
            # 左下: 去背結果（原始值）
            axes[1, 0].imshow(raw_visual, cmap='gray', vmin=0, vmax=255)
            axes[1, 0].set_title('去背結果（原始差值）\n中灰(128)=無差異', fontsize=12)
            axes[1, 0].axis('off')
            
            # 右下: 去背結果（+128）
            axes[1, 1].imshow(visual_result, cmap='gray', vmin=0, vmax=255)
            axes[1, 1].set_title('去背結果（視覺化版本）\n中灰(128)=無差異', fontsize=12)
            axes[1, 1].axis('off')
            
            # 添加說明
            fig.text(0.5, 0.02,
                    f'驗證集 | 方法: {self.background_method.upper()} | '
                    f'左下: 原始差值 | 右下: +128視覺化',
                    ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(debug_dir, f"comparison_{self.background_method}_{base_name}.png"),
                dpi=100, bbox_inches='tight'
            )
            plt.close()
            
            print(f"✅ [VAL DEBUG] 保存debug圖片: {debug_dir}/")
            print(f"   - 原始 + 背景 + 去背(原始) + 去背(視覺化)")
            
        except Exception as e:
            print(f"⚠️ [VAL DEBUG] 保存失敗: {e}")

    def get_image_cache(self, path):
        try:
            return np.load(path)
        except Exception as e:
            raise Exception("File corrupted: " + path)

    def __preprocess_csv(self, csv_file, fps, head_width_px):
        return preprocess_csvV5(csv_file, fps, head_width_px)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        # Load images and convert them to tensors

        img = self.get_image_cache(d['cache_npy'])

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(d['target'])

        img_files = [f"{self.root_dir}/{d['match_name']}/frame/{d['video_name']}/{im}" for im in d['img_files']]

        return {"img": img, "target": target, "img_files": img_files}

    def transform_coordinates(self, data, w, h, target_size=640):
        """
        Transform the X, Y coordinates in data based on image resizing and padding.
        
        Parameters:
        - data (torch.Tensor): A tensor of shape (N, 6) with columns (Frame, Visibility, X, Y, dx, dy).
        - w (int): Original width of the image.
        - h (int): Original height of the image.
        - target_size (int): The desired size for the longest side after resizing.
        
        Returns:
        - torch.Tensor: A transformed tensor of shape (N, 6).
        """
        
        # Clone the data to ensure we don't modify the original tensor in-place
        data_transformed = data
        
        # Determine padding
        max_dim = max(w, h)
        pad_diff = max_dim - min(w, h)
        pad1 = pad_diff // 2
        
        # Indices where x and y are not both 0
        indices_to_transform = (data[:, 2] != 0) | (data[:, 3] != 0)
        
        # Adjust for padding
        if h < w:
            data_transformed[indices_to_transform, 3] += pad1
            data_transformed[indices_to_transform, 5] += pad1
        else:
            data_transformed[indices_to_transform, 2] += pad1  # if height is greater, adjust X
            data_transformed[indices_to_transform, 4] += pad1  # if height is greater, adjust X

        # Adjust for scaling
        scale_factor = target_size / max_dim
        data_transformed[:, 2] *= scale_factor  # scale X
        data_transformed[:, 3] *= scale_factor  # scale Y
        data_transformed[:, 4] *= scale_factor  # scale dx
        data_transformed[:, 5] *= scale_factor  # scale dy
        
        return data_transformed

    def display_image_with_coordinates(self, img_tensor, coordinates):
        """
        Display an image with annotated coordinates.

        Parameters:
        - img_tensor (torch.Tensor): The image tensor of shape (C, H, W) or (H, W, C).
        - coordinates (list of tuples): A list of (X, Y) coordinates to be annotated.
        """
        
        # Convert the image tensor to numpy array
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Create a figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img_array, cmap='gray')

        # Plot each coordinate
        for (x, y) in coordinates:
            ax.scatter(x, y, s=50, c='red', marker='o')

        plt.show()

    @lru_cache(maxsize=10)
    def __preprocess_img(self, path, pad_value=0):
        img = self.open_image(path)
        img = self.pad_to_square(img, pad_value)
        img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        img.resize((1, 640, 640))
        return img

    def open_image(self, path):
        """Open image file, convert to grayscale and resize to half size
        """

        # Open the image file
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        
        # Reduce the resolution to half
        h, w = img.shape

        return img
        
    def pad_to_square(self, img, pad_value=0):
        """Adjust 2D tensor to square by padding {pad_value}
        """
        h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h > w else (pad1, pad2, 0, 0)
        # Add padding
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=pad_value)

        return img
    
if __name__ == '__main__':
    dataset = TrackNetValConfigurableDataset(
        root_dir='/usr/src/datasets/tracknet/val_data',
        num_input=10,
        prefix='[VAL]'
    )