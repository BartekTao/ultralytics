yolov8
    ultralytics\nn\modules\head.py
        調整 detect 輸出，減少長寬以及增加x,y變量

    loss function 修改

    ultralytics\yolo\cfg 修改

    training 的 output weight 要注意 要加重

可以從 dataloader 知道 10 張圖片怎麼丟進去
train_data
    match1
        video
            xxx.mp4
        frame
            frame<ID>.png
        ball_trajectory
            csv file: frame,x,y
    match2
        video
            xxx.mp4
        frame
            frame<ID>.png
        ball_trajectory
            csv file: frame,x,y


程式碼解析
    nn.module parse_model
        解析 models.yaml 轉換成一層一層的網路，輸入的 channel 可以用參數 ch 帶入
    yaml 的最後一層，會接到 Detect 模型，目前需要修改這邊，讓他可以輸出我要的 10 組(x,y,dx,dy,conf)
    data 的部分怎麼塞進來，也是一大問題

    loss
        FL(pi​,ti​)=−αti​​(1−pi​)γlog(pi​) 球存在
        FL(pi​,ti​)=−αti​​piγ​log(1−pi​)   球不存在

## idea
- 每個網格的x,y 大小會有限制 (需要Sigmoid)
- from yoloX: 可以把 cell 中心四周的 cell (3*3)也當作 positive 去算 loss
- 是否可以使用 upsampling 做一些事情
- Kalman Filtering


## train record
- train4: 10/9 使用 non sigmoid predict run 50 epoch
- train1: 嘗試使用 sigmoid(x, y, dx, dy) 只跑了個數的 epoch
- train6: 使用 weight=100
- train49: weight=100, 16 batch, fix focal loss, epoch 50,commit ba2d8ff
- train71: 11/16 tanh, bce loss, epoch 50 commit bd47c9b
- train79: 圖像化，50 epoch，可用來測試 conf loss
- train81: 圖像化，100 epoch，可用來測試 conf loss
- train83: 圖像化，100 epoch，可用來測試 conf loss，包含其他數字
- C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img5: conf 修正完成
- train148 加入所有 loss 100 epoch
- train181 pos+conf loss 50 epoch (head不拆分) check_training_img9
- train183 pos+conf+mov loss 47 epoch (head不拆分) check_training_img10
- train196 pos+conf+mov loss 100 epoch (head不拆分) check_training_img11, commit: f1aecf0 (lr=0.01, momentum=0.9)
- train226 整理後架構 100 epoch

feats = 三個 tensor[16*144*80*80],[16*144*40*40],[16*144*20*20] 
no=144
reg_max=16
nc=80

## 20240518
- 在 predict 的時候，dataset 不知道為什麼會變成 len(0)
- 確認 predict 有問題，有可能是模型沒有儲存到，因為我拿 training data 去 predict，但是結果與 training 的時候不同，目前想法有以下:
    - 拿 best.pt 繼續 train，看看結果長什麼樣子，如果異常，代表 model 沒有正常儲存
    - 檢查 predict 程式碼哪裡異常

## 20240519
- 拿模型 pt 進行訓練，看看訓練時的數據，是否符合預期，以確認 pt 是否有正確儲存

## 20240830
- dxdy 有加與沒加入，對於模型學習的能力是否有正向影響

## 20240830
- train238 測試 epoch 200 with dxdy
- coachbox 密碼: nol56680

## docker debug
### 空間不足
```
docker container prune

//查看 /var/lib/docker/ 目錄及其子目錄的大小
sudo du -sh /var/lib/docker/

//執行以下命令來查看包含 /var/lib/docker/ 的分區的總空間及可用空間
df -h /var/lib/docker/
docker exec -it 0c3e1511776e /bin/bash

```

## 20240907
- train253 測試 epoch 200 without dxdy
- Huber Loss 或 Smooth L1 Loss

## 20240917
- train263 epoch 200 使用 DFL without dxdy and hit loss

## 20240919
待辦
- 確認 263 有多少個 > 0.5 的 conf
- 加入 IOU loss (Gaussian IOU)

## 20240920
- train264 epoch 200 iou sigma=0.7

## 20240923
- train266 epoch 200, 移除 IOU, penalty 調整 TH = 0.5

## 20240925
- train267 epoch 200, FP penalty = 4000

## 20240929
- train278 epoch 1xx, 改 conf loss 但是太慢 (結果是 Dataset 拿錯)
- train286 epoch 6x, focal loss
- train288 epoch 59, focal loss with FN FP 權重
- train294 epoch 200, focal loss with FN FP 權重

## 20241002
- train302 epoch 200, focal loss with 最外層的 weight fb113b3

## 20241005
- train318 epoch 200, DFL + conf + dxdy: 0382e0d

## 20241006
- train322 epoch 200, DFL + conf + dxdy: eba7aca


ERROR training 最後出現錯誤，待排查
Exception in thread Thread-5 (_pin_memory_loop):
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/opt/conda/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/pin_memory.py", line 51, in _pin_memory_loop
    do_one_step()
  File "/opt/conda/lib/python3.10/site-packages/torch/utils/data/_utils/pin_memory.py", line 28, in do_one_step
    r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
  File "/opt/conda/lib/python3.10/multiprocessing/queues.py", line 122, in get
    return _ForkingPickler.loads(res)
  File "/opt/conda/lib/python3.10/site-packages/torch/multiprocessing/reductions.py", line 307, in rebuild_storage_fd
    fd = df.detach()
  File "/opt/conda/lib/python3.10/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/opt/conda/lib/python3.10/multiprocessing/resource_sharer.py", line 86, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/opt/conda/lib/python3.10/multiprocessing/connection.py", line 508, in Client
    answer_challenge(c, authkey)
  File "/opt/conda/lib/python3.10/multiprocessing/connection.py", line 752, in answer_challenge
    message = connection.recv_bytes(256)         # reject large message
  File "/opt/conda/lib/python3.10/multiprocessing/connection.py", line 216, in recv_bytes
    buf = self._recv_bytes(maxlength)
  File "/opt/conda/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "/opt/conda/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
ConnectionResetError: [Errno 104] Connection reset by peer