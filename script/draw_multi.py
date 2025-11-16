from point import Point
import cv2
import os
import csv
import numpy as np
from collections import defaultdict

# === 1. 讀 CSV 並保留所有 frame、保存 visibility ===
csv_path = '/usr/src/ultralytics/runs/detect/predict41/pickle_ball/csv/InPlayBalls_54min_vid/all.csv'
# csv_path = '/usr/src/ultralytics/runs/detect/predict33/sportxai2025_predict/csv/CameraReader_0/all.csv'
frames = defaultdict(list)

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fid = int(row['Frame'])
        # vis = int(row['Visibility'])
        x   = float(row['X'])
        y   = float(row['Y'])
        # x = x * 2048 / 640
        # y = y * 2048 / 640 - (2048 - 1536) / 2
        x = x * 1920 / 640
        y = y * 1920 / 640 - (1920 - 1080) / 2
        # y = y - (640-480)/2
        p   = Point(fid=fid, x=x, y=y)
        p.visibility = 1
        frames[fid].append(p)

# === 2. 讀影片 & 設定輸出 ===
video_path = '/usr/src/datasets/tracknet/predict_data/pickle_ball/video/InPlayBalls_54min_vid.mp4'
# video_path = '/usr/src/datasets/tracknet/predict_data/sportxai2025_predict/video/CameraReader_0.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 60
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

base = os.path.splitext(os.path.basename(video_path))[0]
out  = cv2.VideoWriter(f'{base}_all_points.mp4', fourcc, 120, (w, h))
# out  = cv2.VideoWriter(f'{base}_filtered.mp4', fourcc, fps, (w, h))

# === 3. 逐幀畫點 ===
history = []   # 用來保存過去幾幀的點清單
frame_idx = 0  # 當前正在處理的幀編號

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # (A) 畫當前幀可見點（紅色實心圓）
    curr_points = frames.get(frame_idx, [])
    for p in curr_points:
        if p.visibility == 1:
            cv2.circle(frame, (int(p.x), int(p.y)), 5, (0, 0, 255), -1)

    # (B) 更新 history，最多保留 11 幀（當前 + 前 10）
    # history.append(curr_points)
    # if len(history) > 11:
    #     history.pop(0)

    # (C) 畫前幀前 1～10 幀的點，透明度逐漸降低（藍色調）
    # for j in range(1, min(len(history), 11)):
    #     past_points = history[-1 - j]
    #     alpha = max(0, 1 - 0.1 * j)
    #     color_intensity = int(255 * alpha)
    #     for p in past_points:
    #         if p.visibility == 1:
    #             cv2.circle(frame, (int(p.x), int(p.y)), 5, (0, 0, color_intensity), -1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print('finish')
