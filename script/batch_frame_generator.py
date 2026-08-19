#!/usr/bin/env python3
import os
import subprocess
import glob
from tqdm import tqdm

# ================= 配置區 =================
# 確保這裡指向包含「時間戳資料夾」的那個 video 大目錄
VIDEO_ROOT = "/usr/src/datasets/tracknet/train_data/pickleball/video"
# 圖片輸出的根目錄
FRAME_ROOT = "/usr/src/datasets/tracknet/train_data/pickleball/frame"
FRAME_GENERATOR = "/usr/src/ultralytics/script/Frame_Generator.py"
# =========================================

# 檢查路徑
if not os.path.exists(VIDEO_ROOT):
    print(f"❌ 錯誤：找不到影片根目錄 {VIDEO_ROOT}")
    exit(1)

# 取得所有時間戳子目錄
match_dirs = sorted([d for d in os.listdir(VIDEO_ROOT) 
                     if os.path.isdir(os.path.join(VIDEO_ROOT, d))])

if not match_dirs:
    print(f"❓ 在 {VIDEO_ROOT} 下沒有找到任何目錄")
    exit(1)

print(f"找到 {len(match_dirs)} 個待處理目錄")
print("=" * 70)

successful_matches = []
failed_matches = []

for match_name in tqdm(match_dirs, desc="Processing"):
    # 影片來源目錄 (例如: .../video/2026-03-30_14-32-17)
    current_video_path = os.path.join(VIDEO_ROOT, match_name)
    
    # 【關鍵修改點】：定義該場次專屬的輸出目錄 (例如: .../frame/2026-03-30_14-32-17)
    # 這樣 Frame_Generator 的 shutil.rmtree 刪除時，只會刪到這個場次的舊資料
    current_frame_output = os.path.join(FRAME_ROOT, match_name)
    
    # 檢查是否有影片檔案，避免空跑
    video_files = glob.glob(os.path.join(current_video_path, "*.mp4")) + \
                  glob.glob(os.path.join(current_video_path, "*.avi"))
    
    if not video_files:
        continue

    try:
        # 參數 1: 影片來源目錄
        # 參數 2: 該場次專屬的輸出目錄 (避開全局刪除邏輯)
        cmd = ['python3', FRAME_GENERATOR, current_video_path, current_frame_output]
        
        # 執行轉換並捕捉結果
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            successful_matches.append(match_name)
        else:
            print(f"\n❌ {match_name} 執行失敗：{result.stderr}")
            failed_matches.append(match_name)
            
    except Exception as e:
        print(f"\n❌ {match_name} 遇到系統異常: {e}")
        failed_matches.append(match_name)

# 最終統計
print("\n" + "=" * 70)
print(f"📊 統計結果: 成功 {len(successful_matches)} | 失敗 {len(failed_matches)}")
print("=" * 70)

if successful_matches:
    print(f"✅ 所有圖片已生成至: {FRAME_ROOT}")