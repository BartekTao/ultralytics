## 環境

- Python `3.10.11`（同 base image `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`）

## dataset

for example：
- in 140.113.208.122 coachbox
    - training data
        - /hdd/dataset/alex_tracknet
        - /hdd/dataset/blion_tracknet
        - /hdd/dataset/profession_match_{n}
    - testing data
        - /hdd/dataset/profession_match_{n}_test

以上是主機端實體路徑，目前是整個 `train_data`/`val_data` 目錄掛進 container（見下方 run 指令），
不是每個 match 分開掛。

**npy 前處理快取**（`/ssd2/tracknet_cache`）是另一顆獨立的 NVMe，不跟資料集共用碟——訓練時每個
batch 都要從這裡隨機讀取大量小檔案，用一般硬碟會直接變成瓶頸。這顆碟也要單獨掛進 container
（見下方 run 指令），不要漏掉，不然每次開新 container 都要重建全部快取。

## how to build

`ULTRALYTICS_BRANCH`：指定使用分支
`CACHE_BUSTER`：確保每次都能拉取最新的分支（用 timestamp 讓 Docker 不要吃到 build cache）

```
docker build \
    -f ultralytics/tracknet/Dockerfile \
    --build-arg ULTRALYTICS_BRANCH=feat/test5555 \
    --build-arg CACHE_BUSTER=$(date +%s) \
    -t tracknet1000:latest .
```

## how to run

```
docker run --gpus all --ipc=host \
-v <主機端資料集路徑>/train_data:/usr/src/datasets/tracknet/train_data \
-v <主機端資料集路徑>/val_data:/usr/src/datasets/tracknet/val_data \
-v <主機端資料集路徑>/val_data/.cache:/usr/src/datasets/tracknet/val_data/.cache \
-v <主機端資料集路徑>/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-v <主機端資料集路徑>/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
-v <主機端 runs 路徑>:/usr/src/ultralytics/runs \
-v <主機端 npy cache 路徑，一定要是 SSD/NVMe>:/ssd2/tracknet_cache \
-it tracknet1000:latest
```

`val_data/.cache` 是舊版 cache 機制（改到 `/ssd2/tracknet_cache` 這顆獨立 NVMe之前）留下來的，目前程式碼的 cache 路徑都寫死指向 `/ssd2/tracknet_cache`。

啟動之後：

```
# 訓練（從頭開始，model 給 .yaml）
python tracknet.py --mode train_v2 \
    --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml \
    --epochs 200 \
    --dataset_config ultralytics/tracknet/dataset_split.json

# Fine-tune (別的 checkpoint)
python tracknet.py --mode train \
    --model_path /usr/src/ultralytics/runs/detect/trainXXX/weights/best.pt \
    --epochs 50 \
    --dataset_config ultralytics/tracknet/dataset_split.json

# 驗證
python tracknet.py --mode val_v2 --batch 1 \
    --model_path /usr/src/ultralytics/runs/detect/train345/weights/last.pt
```

要練哪些資料、各自要抓多少張，改 `ultralytics/tracknet/dataset_split.json`外部設定檔（`{"train": {...}, "val": {...}}`），

`workers` 在 `tracknet.py` 裡寫死 16，但實際生效的數字是 `min(os.cpu_count() // GPU 數, 16)`；
在有限制 CPU 配額的 container 裡，`os.cpu_count()` 不一定準（可能回報主機的實體核心數而非
container 實際配額），建議進 container 後跑一次 `nproc` 對一下，數字差太多就手動調整。