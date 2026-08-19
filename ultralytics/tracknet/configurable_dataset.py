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
from ultralytics.tracknet.utils.preprocess import preprocess_csv, preprocess_csvV5, load_path_counts

class TrackNetConfigurableDataset(Dataset):

    def __init__(self, root_dir, num_input=10, transform=None, prefix='',
                 background_method='mean', use_downsample=True, ds_min_fps=30, ds_maxstep=2,
                 dataset_config=None):

        self.match_mog2 = {}
        self.root_dir = root_dir
        self.transform = transform
        self.num_input = num_input
        self.samples = []
        self.prefix = prefix
        self.use_downsample = use_downsample
        self.ds_min_fps = ds_min_fps
        self.ds_maxstep = ds_maxstep

        # ============ background removal setup ============
        self.background_method = background_method
        print(f"\n{'='*60}")
        print(f"[背景去除配置] Method: {self.background_method.upper()}")
        print(f"{'='*60}\n")

        # ====================================
        # self.path_counts = {f"profession_match_{i}": 1000 for i in range(1, 30)}
        # self.path_counts.update({
        #     "match_2": 5000, # for local test
        #     # "AUX_nycu_new_court": 2000,
        #     # "nycu_new_court_2048_1536": 2000,
        #     # "sportxai_serve_machine": 2000,
        #     # "sportxai_rally": 2000,
        #     # "hsinchu_gym": 2000,
        #     # "ces2025_all": 2000,
        #     # "office_dataset": 2000,
        # })
        # self.path_counts = {
        #     "profession_game" : 10000,
        #     "AUX_nycu_new_court": 2000,
        #     "BUX_nycu_new_court": 2000,
        #     "meichu_new_court": 2000,
        #     "sportxai_serve_machine": 2000,
        #     "sportxai_rally": 2000,
        #     "hsinchu_gym": 2000,    
        #     "ITRI_CES_data": 2000,
        #     "office_dataset": 2000,
        #     "nctu_old_gym": 2000,
        #     "sport114" : 2000,
        #     "hsinchu_old_gym": 2000,
        #     "national_ranking_114": 2000,
        #     "green_wall": 1000,
        #     "green_wall_2": 1000,
        #     "blue_wall" : 1000,
        #     "EC234": 1000,
        #     "EC_4F_Corridor": 1000,
        #     "EC330": 1000
        # }

        """ train115&117
        self.path_counts = {
            "sportxai_serve_machine": 3000,
            "sportxai_rally": 3000,
            "sportxai_2025": 4000,
            "profession_game_dataset_others": 10000
        }
        """
        """self.path_counts = {
            "profession_game" : 10000,
            "AUX_nycu_new_court": 2000,
            "BUX_nycu_new_court": 2000,
            "meichu_new_court": 2000,
            "sportxai_serve_machine": 2000,
            "sportxai_rally": 2000,
            "hsinchu_gym": 2000,    
            "ITRI_CES_data": 2000,
            "office_dataset": 2000,
            "nctu_old_gym": 2000,
            "sport114" : 2000,
            "hsinchu_old_gym": 2000,
            "national_ranking_114": 2000,
            "green_wall": 1000,
            "green_wall_2": 1000,
            "blue_wall" : 1000,
            "EC234": 1000,
            "EC_4F_Corridor": 1000,
            "EC330": 1000,
            "profession_game_dataset_others": 10000
        }"""

        # 資料集選擇改由外部檔案管理，見 dataset_split.json 的 "train" 區塊；
        # 找不到檔案/沒傳 dataset_config 時退回這個內建預設值。
        self.path_counts = load_path_counts(dataset_config, "train", fallback={"pickleball": 10000})

        self.idx = set()
        self.match_sample_counts = {}   # 追蹤每個 match 的實際樣本數（含 hit 擴充）

        image_count = len(glob(os.path.join(self.root_dir, "*/", "frame/", "*/", "*.png")))

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
            
            # 支援多層目錄結構：如果某個目錄下有 metadata.json 和子目錄（match1, match2 等）
            self._process_nested_matches(match_name, match_dir_path)

        # 統計每個 match 的實際樣本數（含 hit 擴充後）
        from collections import Counter
        self.match_sample_counts = dict(Counter(s["match_name"] for s in self.samples))


    def get_dataset_config(self) -> dict:
        """回傳這個 dataset 的完整配置，供訓練結束後輸出 JSON 用"""
        return {
            "root_dir": self.root_dir,
            "num_input_frames": self.num_input,
            "background_method": self.background_method,
            "use_downsample": self.use_downsample,
            "ds_min_fps": self.ds_min_fps,
            "ds_maxstep": self.ds_maxstep,
            "input_size": "640x640",
            "total_samples": len(self.samples),
            "path_counts_config": dict(self.path_counts),
            "match_sample_counts": self.match_sample_counts,
            "match_dirs": sorted(self.match_sample_counts.keys()),
        }

    # 支援多層目錄結構：如果某個目錄下有 metadata.json 和子目錄（match1, match2 等），則對每個子目錄分別處理，並根據 path_counts 分配樣本數量
    def _process_nested_matches(self, parent_name, parent_dir):
        if parent_name not in self.path_counts:
            return
        
        metadata_path = os.path.join(parent_dir, 'metadata.json')
        if not os.path.isfile(metadata_path):
            return
        
        if all(os.path.isdir(os.path.join(parent_dir, d)) for d in ['video', 'csv', 'frame']):
            return
        
        sub_matches = [d.strip('/') for d in sorted(glob("*/", root_dir=parent_dir))
                       if os.path.isdir(os.path.join(parent_dir, d.strip('/')))]
        
        valid_sub_matches = []
        for sub_match_name in sub_matches:
            sub_match_dir = os.path.join(parent_dir, sub_match_name)
            if all(os.path.isdir(os.path.join(sub_match_dir, d)) for d in ['video', 'csv', 'frame']):
                valid_sub_matches.append(sub_match_name)
        
        if not valid_sub_matches:
            return
        
        total_limit = self.path_counts[parent_name]
        limit_per_match = max(1, total_limit // len(valid_sub_matches))
        
        last_len = len(self.samples)
        for sub_match_name in valid_sub_matches:
            sub_match_dir = os.path.join(parent_dir, sub_match_name)
            
            try:
                image_count = len(glob(os.path.join(sub_match_dir, "frame/", "*/", "*.png")))
                total_samples = image_count
                
                full_match_name = f"{parent_name}/{sub_match_name}"
                
                with tqdm(total=total_samples, desc=f"Processing {full_match_name}", miniters=1, smoothing=1) as pbar:
                    self.read_match_nested(full_match_name, parent_dir, sub_match_dir, pbar, limit_per_match)
                
                print(f"Total samples for {full_match_name}: {len(self.samples)-last_len}\n")
                last_len = len(self.samples)
            except Exception as e:
                print(f"Warning: Failed to process {full_match_name}: {e}\n")
                continue

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

            limit_count = self.path_counts[match_name]
            samples_added_count = 0

            # Traverse all videos in the match directory
            for video_file in video_files:
                if samples_added_count >= limit_count:
                    break

                video_path = os.path.join(video_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                cap.release()

                # base name without extension
                video_base, _ = os.path.splitext(video_file)

                csv_file = os.path.join(csv_dir, video_base + "_ball" + '.csv')
                if not os.path.isfile(csv_file):
                    # 如果沒有對應的 csv，跳過（或可改成 warning）
                    # print(f"Warning: missing csv for {video_base}, skip")
                    continue

                ball_trajectory_df = self.__preprocess_csv(csv_file, fps, head_width)

                frame_dir = os.path.join(self.root_dir, match_name, 'frame', video_base)
                if not os.path.isdir(frame_dir):
                    continue

                # 更穩健的 png 檔名排序（避免 removesuffix 在某些 Py 版本問題）
                img_files = sorted(glob("*.png", root_dir=frame_dir),
                                   key=lambda x: int(os.path.splitext(x)[0]))
                total_img_len = len(img_files)
                min_len = min(limit_count - samples_added_count, total_img_len)
                if min_len == 0:
                    continue

                img = cv2.imread(os.path.join(frame_dir, img_files[0]))
                if img is None:
                    continue
                height, width, _ = img.shape

                # Create sliding windows of num_input frames
                for i in range(min_len - (self.num_input-1)):
                    if samples_added_count >= limit_count:
                        break

                    frames = img_files[i: i + self.num_input]

                    target = ball_trajectory_df.iloc[i: i + self.num_input].values
                    target = self.transform_coordinates(target, width, height)

                    # Avoid invalid data
                    if len(frames) == self.num_input and len(target) == self.num_input:
                        frame_cache_paths = self.img_cache_dir(match_name, video_base, frames)

                        self.samples.append({
                            "match_name": match_name,
                            "video_name": video_base,
                            "frame_cache_paths": frame_cache_paths,
                            "img_files": frames,
                            "target": target
                        })

                        for fp, cache_path in zip(frames, frame_cache_paths):
                            self.img_cache_frame(frame_dir, fp, cache_path)

                        samples_added_count += 1

                        hit_exists = np.any(target[:, 6] == 1)
                        if hit_exists:
                            for _ in range(5):
                                self.samples.append({
                                    "match_name": match_name,
                                    "video_name": video_base,
                                    "frame_cache_paths": frame_cache_paths,
                                    "img_files": frames,
                                    "target": target
                                })

                # min_fps = 50
                # print(fps, min_fps)
                # valid_steps = self.get_valid_downsample_steps(fps, min_fps)
                # print(valid_steps)
                # valid_steps = [2]

                if self.use_downsample:
                    valid_steps = self.get_valid_downsample_steps(fps, self.ds_min_fps, self.ds_maxstep)
                else:
                    valid_steps = []

                for step in valid_steps:
                    num_frames_needed = self.num_input * step
                    max_start_idx = len(img_files) - num_frames_needed + 1

                    for i in range(max_start_idx):
                        if samples_added_count >= limit_count:
                            break

                        frames = img_files[i: i + num_frames_needed: step]
                        target = ball_trajectory_df.iloc[i: i + num_frames_needed: step].values
                        target = self.transform_coordinates(target, width, height)

                        if len(frames) == self.num_input and len(target) == self.num_input:
                            frame_cache_paths = self.img_cache_dir(match_name, video_base, frames)

                            sample = {
                                "match_name": match_name,
                                "video_name": video_base,
                                "frame_cache_paths": frame_cache_paths,
                                "img_files": frames,
                                "target": target
                            }

                            self.samples.append(sample)
                            for fp, cache_path in zip(frames, frame_cache_paths):
                                self.img_cache_frame(frame_dir, fp, cache_path)

                            samples_added_count += 1

                            # 擴充 hit 樣本
                            if np.any(target[:, 6] == 1):
                                for _ in range(5):
                                    self.samples.append(sample.copy())

                pbar.update(min_len)

            self.path_counts[match_name] = limit_count - samples_added_count

    def get_valid_downsample_steps(self, original_fps: int, min_fps: int, max_step: int = None) -> list[int]:
        steps = [step for step in range(2, original_fps + 1) if original_fps / step >= min_fps]
        if max_step is not None:
            steps = [step for step in steps if step <= max_step]
        return steps

    def img_cache_dir(self, match_name, video_name, img_files):
        """回傳這個 10 幀窗對應的每一幀快取路徑（list），並用窗的身分（match+video+frame 檔名序列）
        偵測是否重複處理到同一個窗——偵測邏輯與舊版相同，只是快取本身改成以幀為單位存放
        （見 frame_cache_path / img_cache_frame），避免相鄰視窗因 stride=1 重疊 9 幀而重複
        decode/resize/存檔近 10 次。
        """
        s = '|'.join([match_name]+[video_name]+img_files)
        filename = hashlib.sha1(s.encode('utf-8')).hexdigest()

        if filename in self.idx:
            raise Exception('DUP: '+filename)
        self.idx.add(filename)

        return [self.frame_cache_path(match_name, video_name, fp) for fp in img_files]

    def frame_cache_path(self, match_name, video_name, img_file):
        s = '|'.join([match_name, video_name, img_file])
        filename = hashlib.sha1(s.encode('utf-8')).hexdigest()

        cache_base = "/ssd2/tracknet_cache/train_data" if "train_data" in self.root_dir else "/ssd2/tracknet_cache/val_data"
        d = os.path.join(cache_base, self.background_method, "frames", filename[:2], filename[2:4])

        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{filename}.npy")

    def img_cache_frame(self, frame_dir, img_file, frame_npy_path):
        """快取單一幀：灰階 + pad-to-square + resize。不含背景相減——背景相減依賴同一個窗內
        其他幀，是窗層級的運算，留到 assemble_window() 在讀取時才做。"""
        if os.path.isfile(frame_npy_path):
            return

        img = cv2.imread(os.path.join(frame_dir, img_file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = self.pad_to_square(img)
        img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        np.save(frame_npy_path, img)

    def assemble_window(self, frame_cache_paths):
        """讀取一個窗的 10 個幀快取，組成 [10,640,640]，並依 background_method 做背景相減。"""
        frames = np.stack([self.get_image_cache(p) for p in frame_cache_paths], axis=0)

        if self.background_method == 'none':
            processed_frames = frames
        elif self.background_method == 'median':
            bg_frame = np.median(frames, axis=0).astype(np.float32)
            processed_frames = (frames - bg_frame).astype(np.float32)
        elif self.background_method == 'mean':
            bg_frame = np.mean(frames, axis=0).astype(np.float32)
            processed_frames = (frames - bg_frame).astype(np.float32)
        else:
            raise ValueError(
                f"未知的 background_method: '{self.background_method}'\n"
                f"有效選項: 'none', 'median', 'mean'"
            )

        return processed_frames.astype(np.float32)

    # v2 版本的影像快取，使用 MOG2 背景減除法，測試效果較差
    def img_cache_v2(self, match_name, video_name, img_files, npy_path):
        if os.path.isfile(npy_path):
            return

        # 確保該 match_name 有專屬的 MOG2
        if match_name not in self.match_mog2:
            self.match_mog2[match_name] = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
        mog2 = self.match_mog2[match_name]

        images = []

        for fp in img_files:
            img_path = os.path.join(self.root_dir, match_name, 'frame', video_name, fp)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_float = img.astype(np.float32)

            # 前景提取
            fg_mask = mog2.apply(img_float)

            # 用 mask 取得灰階前景
            foreground = cv2.bitwise_and(img_float, img_float, mask=fg_mask)

            # pad_to_square & resize
            img_square = self.pad_to_square(foreground)
            img_resized = cv2.resize(img_square, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)

            # expand dims
            img_exp = np.expand_dims(img_resized, axis=0)
            images.append(img_exp)

        # 合併所有 frames
        img_stack = np.concatenate(images, axis=0)
        np.save(npy_path, img_stack)

    def get_image_cache(self, path):
        try:
            return np.load(path)
        except Exception as e:
            raise Exception("File corrupted: " + path)

    def __preprocess_csv(self, csv_file, fps, head_width_px):
        return preprocess_csvV5(csv_file, fps, head_width_px)
        # return preprocess_csv(csv_file)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = self.samples[idx]
        # Load images and convert them to tensors

        img = self.assemble_window(d['frame_cache_paths'])

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
            ## Optionally, you can also draw a small rectangle around each point
            #rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=1, edgecolor='red', facecolor='none')
            #ax.add_patch(rect)

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
    def read_match_nested(self, full_match_name, parent_dir, sub_match_dir, pbar, limit_per_match=None):
        """
        處理嵌套的 match 結構
        使用父目錄的 metadata.json，但從子目錄讀取 video/csv/frame
        
        參數：
        - full_match_name: "parent/sub_match" (用於 sample 記錄)
        - parent_dir: 父目錄路徑（metadata.json 所在位置）
        - sub_match_dir: 子目錄路徑（video/csv/frame 所在位置）
        - pbar: 進度條
        - limit_per_match: 此 match 應該取的樣本數上限（平分後的值）
        """
        # 從父目錄讀取 metadata
        metadata_path = os.path.join(parent_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        head_width = data['calibration']['near_camera_head_width_px']

        # 從子目錄讀取 video/csv/frame
        video_dir = os.path.join(sub_match_dir, 'video')
        csv_dir = os.path.join(sub_match_dir, 'csv')

        pbar.set_description(f'{self.prefix} Generating image cache: {full_match_name}/ ')

        # 獲取父目錄名稱用於 path_counts 檢查
        parent_name = os.path.basename(parent_dir)
        
        # 決定使用的 limit
        if limit_per_match is None:
            limit_per_match = self.path_counts.get(parent_name, float('inf'))
        
        if parent_name in self.path_counts:
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

                frame_dir = os.path.join(sub_match_dir, 'frame', video_base)
                if not os.path.isdir(frame_dir):
                    continue

                # 更穩健的 png 檔名排序
                img_files = sorted(glob("*.png", root_dir=frame_dir),
                                   key=lambda x: int(os.path.splitext(x)[0]))
                total_img_len = len(img_files)
                # 使用平分後的限制數量
                min_len = min(limit_per_match, total_img_len)
                if min_len == 0:
                    continue

                img = cv2.imread(os.path.join(frame_dir, img_files[0]))
                if img is None:
                    continue
                height, width, _ = img.shape
                
                for i in range(min_len - (self.num_input-1)):
                    if samples_added_count >= limit_per_match:
                        break
                    
                    frames = img_files[i: i + self.num_input]

                    target = ball_trajectory_df.iloc[i: i + self.num_input].values
                    target = self.transform_coordinates(target, width, height)

                    # Avoid invalid data
                    if len(frames) == self.num_input and len(target) == self.num_input:
                        frame_cache_paths = self.img_cache_dir(full_match_name, video_base, frames)

                        self.samples.append({
                            "match_name": full_match_name,
                            "video_name": video_base,
                            "frame_cache_paths": frame_cache_paths,
                            "img_files": frames,
                            "target": target
                        })

                        for fp, cache_path in zip(frames, frame_cache_paths):
                            self.img_cache_frame(frame_dir, fp, cache_path)

                        hit_exists = np.any(target[:, 6] == 1)
                        if hit_exists:
                            for _ in range(5):
                                self.samples.append({
                                    "match_name": full_match_name,
                                    "video_name": video_base,
                                    "frame_cache_paths": frame_cache_paths,
                                    "img_files": frames,
                                    "target": target
                                })

                        pbar.update(1)
                        
                        samples_added_count += 1

                if self.use_downsample:
                    valid_steps = self.get_valid_downsample_steps(fps, self.ds_min_fps, self.ds_maxstep)
                else:
                    valid_steps = []

                for step in valid_steps:
                    num_frames_needed = self.num_input * step
                    max_start_idx = len(img_files) - num_frames_needed + 1

                    for i in range(max_start_idx):
                        if samples_added_count >= limit_per_match:
                            break
                        
                        frames = img_files[i: i + num_frames_needed: step]
                        target = ball_trajectory_df.iloc[i: i + num_frames_needed: step].values
                        target = self.transform_coordinates(target, width, height)

                        if len(frames) == self.num_input and len(target) == self.num_input:
                            frame_cache_paths = self.img_cache_dir(full_match_name, video_base, frames)

                            sample = {
                                "match_name": full_match_name,
                                "video_name": video_base,
                                "frame_cache_paths": frame_cache_paths,
                                "img_files": frames,
                                "target": target
                            }

                            self.samples.append(sample)
                            for fp, cache_path in zip(frames, frame_cache_paths):
                                self.img_cache_frame(frame_dir, fp, cache_path)

                            # 擴充 hit 樣本
                            hit_exists = np.any(target[:, 6] == 1)
                            if hit_exists:
                                for _ in range(5):
                                    self.samples.append(sample)

                            pbar.update(1)
                            
                            # 計數基礎樣本（不計擴充）
                            samples_added_count += 1