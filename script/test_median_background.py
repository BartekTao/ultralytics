import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取 10 張圖片

file_paths = [
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/80.png", 
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/81.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/82.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/83.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/84.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/85.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/86.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/87.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/88.png",
    "/usr/src/datasets/tracknet/train_data/profession_game/frame/1_01_01/89.png",
]

frames = [cv2.imread(fp, cv2.IMREAD_GRAYSCALE) for fp in file_paths]


# 計算中位數影像
median_frame = np.median(frames, axis=0).astype(np.float32)

# 影像去除中位數
processed_frames = []
for f in frames:
    diff = f.astype(np.float32) - median_frame      # 負 => 比背景暗, 正 => 比背景亮
    out = np.clip(diff + 128.0, 0, 255).astype(np.uint8)  # 0 的差對應到 128 灰
    processed_frames.append(out)

# 顯示部分原始影像、處理後影像和中位數影像
fig, axes = plt.subplots(3, 10, figsize=(15, 9))

for i in range(10):
    # 原始影像
    axes[0, i].imshow(frames[i], cmap='gray')
    axes[0, i].set_title(f'Original Frame {i+1}')
    axes[0, i].axis('off')

    # 去除中位數後的影像
    axes[1, i].imshow(processed_frames[i], cmap='gray')
    axes[1, i].set_title(f'Processed Frame {i+1}')
    axes[1, i].axis('off')

# 顯示計算出的中位數影像
axes[2, 2].imshow(median_frame, cmap='gray')
axes[2, 2].set_title('Median Frame')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()

cv2.imwrite("/usr/src/ultralytics/script/test_background/median_frame.png", median_frame.astype(np.uint8))
for i, processed_frame in enumerate(processed_frames):
    output_path = f"/usr/src/ultralytics/script/test_background/processed_median_{i+1}.png"
    cv2.imwrite(output_path, processed_frame)
