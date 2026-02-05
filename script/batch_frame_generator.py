#!/usr/bin/env python3
"""
批次生成所有 match 的 frame
用於 profession_game_dataset_others/match1, match2, ... match29
"""

import os
import sys
import subprocess
import glob
from pathlib import Path
from tqdm import tqdm

# 配置
DATASET_ROOT = "/usr/src/datasets/tracknet/train_data"
PARENT_DIR = "profession_game_dataset_others"
FRAME_GENERATOR = "/usr/src/ultralytics/script/Frame_Generator.py"

# 完整路徑
parent_path = os.path.join(DATASET_ROOT, PARENT_DIR)

# 檢查父目錄是否存在
if not os.path.isdir(parent_path):
    print(f"❌ 錯誤：目錄不存在 {parent_path}")
    sys.exit(1)

# 檢查 Frame_Generator 腳本是否存在
if not os.path.isfile(FRAME_GENERATOR):
    print(f"❌ 錯誤：找不到 Frame_Generator 腳本 {FRAME_GENERATOR}")
    sys.exit(1)

# 掃描所有 match 目錄 (match1, match2, ...)
match_dirs = sorted([d for d in os.listdir(parent_path) 
                     if os.path.isdir(os.path.join(parent_path, d)) 
                     and d.startswith('match')])

if not match_dirs:
    print(f"❌ 錯誤：在 {parent_path} 中找不到任何 match 目錄")
    sys.exit(1)

print(f"找到 {len(match_dirs)} 個 match 目錄")
print(f"目錄列表: {match_dirs}")
print("=" * 70)

# 逐個處理每個 match
failed_matches = []
successful_matches = []

for match_name in tqdm(match_dirs, desc="Processing matches"):
    match_path = os.path.join(parent_path, match_name)
    video_dir = os.path.join(match_path, 'video')
    frame_dir = os.path.join(match_path, 'frame')
    
    # 檢查 video 目錄是否存在
    if not os.path.isdir(video_dir):
        print(f"\n⚠️  跳過 {match_name}：video 目錄不存在")
        failed_matches.append(match_name)
        continue
    
    # 檢查是否影片檔案
    video_files = glob.glob(os.path.join(video_dir, '*.mp4')) + \
                  glob.glob(os.path.join(video_dir, '*.avi'))
    
    if not video_files:
        print(f"\n⚠️  跳過 {match_name}：video 目錄中沒有 .mp4 或 .avi 檔案")
        failed_matches.append(match_name)
        continue
    
    # 執行 Frame_Generator
    try:
        print(f"\n▶️  正在處理 {match_name} ({len(video_files)} 個影片)...")
        cmd = ['python3', FRAME_GENERATOR, video_dir, frame_dir]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)  # 24小時超時
        
        if result.returncode == 0:
            # 檢查 frame 目錄是否成功創建
            if os.path.isdir(frame_dir):
                frame_count = sum([len(files) for _, _, files in os.walk(frame_dir)])
                print(f"✅ {match_name} 完成 (生成 {frame_count} 個 frame)")
                successful_matches.append(match_name)
            else:
                print(f"❌ {match_name} 失敗：frame 目錄未創建")
                failed_matches.append(match_name)
        else:
            print(f"❌ {match_name} 失敗")
            print(f"   錯誤: {result.stderr}")
            failed_matches.append(match_name)
            
    except subprocess.TimeoutExpired:
        print(f"❌ {match_name} 超時（超過 24 小時）")
        failed_matches.append(match_name)
    except Exception as e:
        print(f"❌ {match_name} 異常: {e}")
        failed_matches.append(match_name)

# 最終統計
print("\n" + "=" * 70)
print("📊 最終統計")
print("=" * 70)
print(f"✅ 成功: {len(successful_matches)}/{len(match_dirs)}")
if successful_matches:
    for m in successful_matches:
        print(f"   - {m}")

if failed_matches:
    print(f"\n❌ 失敗: {len(failed_matches)}/{len(match_dirs)}")
    for m in failed_matches:
        print(f"   - {m}")

if len(successful_matches) == len(match_dirs):
    print("\n🎉 所有 match 都成功處理！")
    sys.exit(0)
else:
    print(f"\n⚠️  {len(failed_matches)} 個 match 處理失敗")
    sys.exit(1)
