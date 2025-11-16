"""
TrackNet Video Dataset (流式處理版本)
不預載所有幀，節省記憶體
"""

import os
import time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

from ultralytics.yolo.data.build import check_source
from ultralytics.yolo.data.dataloaders.stream_loaders import SourceTypes


class TrackNetVideoDataset(Dataset):
    """
    從影片文件讀取數據的 Dataset（流式處理版本）
    
    Args:
        video_path: 影片文件路徑
        num_input: 每個 batch 的幀數（默認 10）
        imgsz: 輸入圖像大小（默認 640）
        stride: 滑動窗口步長（默認 10，不重疊）
        save_raw_frames: 是否保存原始幀到硬碟
        raw_frame_dir: 原始幀保存目錄
        transform: 數據轉換（可選）
    """
    
    def __init__(self, video_path, num_input=10, imgsz=640, stride=10, 
                 save_raw_frames=False, raw_frame_dir=None, transform=None):
        self.video_path = video_path
        self.num_input = num_input
        self.imgsz = imgsz
        self.stride = stride
        self.transform = transform
        self.save_raw_frames = save_raw_frames
        self.raw_frame_dir = raw_frame_dir
        self.bs = 1
        
        # 打開影片獲取基本信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        
        # 影片屬性
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Video info: {self.total_frames} frames, {self.fps:.2f} FPS, {self.width}x{self.height}")
        
        # 計算總 batch 數
        self.total_batches = (self.total_frames - self.num_input) // self.stride + 1
        print(f"Total batches: {self.total_batches} (stride={self.stride})")
        
        # 設置 source_type
        source, webcam, screenshot, from_img, in_memory, tensor = check_source(video_path)
        self.source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)
        
        # 如果需要保存原始幀，預先創建目錄
        if self.save_raw_frames and self.raw_frame_dir:
            os.makedirs(self.raw_frame_dir, exist_ok=True)
    
    def __len__(self):
        return self.total_batches
    
    def __getitem__(self, idx):
        """
        即時讀取指定 batch 的數據
        
        Returns:
            path: 幀標識
            img_tensor: (num_input, H, W) 灰階張量
            frames_color: 長度為 num_input 的彩色幀列表
            vid_cap: 空字符串
            fids: 幀 ID 列表
            timestamps: 時間戳列表
        """
        start_frame = idx * self.stride
        end_frame = start_frame + self.num_input
        
        # 打開影片並跳到指定幀
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        batch_frames_gray = []
        batch_frames_color = []
        fids = []
        timestamps = []
        
        # 讀取 num_input 幀
        for i in range(self.num_input):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {start_frame + i}")
                break
            
            current_fid = start_frame + i
            
            # 保存彩色幀
            batch_frames_color.append(frame.copy())
            
            # 轉換為灰階並預處理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            gray = self.pad_to_square(gray)
            gray = cv2.resize(gray, dsize=(self.imgsz, self.imgsz), 
                            interpolation=cv2.INTER_CUBIC)
            gray = np.expand_dims(gray, axis=0)  # (1, H, W)
            batch_frames_gray.append(gray)
            
            fids.append(current_fid)
            timestamps.append(time.time())
            
            # 保存原始幀（如果需要）
            if self.save_raw_frames and self.raw_frame_dir:
                frame_path = os.path.join(self.raw_frame_dir, f'{current_fid}.png')
                cv2.imwrite(frame_path, frame)
        
        cap.release()
        
        # 拼接成張量
        if len(batch_frames_gray) == self.num_input:
            img = np.concatenate(batch_frames_gray, 0)  # (num_input, H, W)
            
            if self.transform:
                img = self.transform(img)
            
            img_tensor = torch.from_numpy(img).float()
            
            return (
                f"frame_{start_frame}",
                img_tensor,
                batch_frames_color,
                "",
                fids,
                timestamps
            )
        else:
            # 如果讀取失敗，返回空數據
            print(f"Warning: Incomplete batch at index {idx}")
            return None
    
    def pad_to_square(self, img, pad_value=0):
        """將圖像填充為正方形"""
        h, w = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if h > w else (pad1, pad2, 0, 0)
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=pad_value)
        return img
    
    def get_video_info(self):
        """返回影片基本信息"""
        return {
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_batches': self.total_batches
        }