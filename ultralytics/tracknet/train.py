from ultralytics.tracknet.configurable_dataset import TrackNetConfigurableDataset
from ultralytics.tracknet.val_configurable_dataset import TrackNetValConfigurableDataset
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.tracknet_v4 import TrackNetV4Model
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.tracknet.val_dataset import TrackNetValDataset
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from copy import copy
from torch.utils.data import random_split
import torch
import json
import os
from datetime import datetime

class TrackNetTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        # generator = torch.Generator().manual_seed(42)
        # dataset = TrackNetConfigurableDataset(root_dir=img_path)
        # train_size = int(0.8 * len(dataset))  # 70% 的數據作為訓練集
        # val_size = len(dataset) - train_size  # 剩下的 30% 作為驗證集
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

        background_method = getattr(self.args, 'background_method', 'mean')
        use_downsample = getattr(self.args, 'use_downsample', True)
        ds_min_fps = getattr(self.args, 'ds_min_fps', 30)
        ds_maxstep = getattr(self.args, 'ds_maxstep', 2)
        dataset_config = getattr(self.args, 'dataset_config', None)

        if mode == 'train':
            dataset = TrackNetConfigurableDataset(root_dir=img_path, background_method=background_method, use_downsample=use_downsample, ds_min_fps=ds_min_fps, ds_maxstep=ds_maxstep, dataset_config=dataset_config)
            return dataset
        else:
            dataset = TrackNetValConfigurableDataset(root_dir=img_path, background_method=background_method, dataset_config=dataset_config)
            #dataset = TrackNetValDataset(root_dir=img_path)
            return dataset

    def get_model(self, cfg=None, weights=None, verbose=True):
        self.tracknet_model = TrackNetV4Model(cfg, ch=10, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            self.tracknet_model.load(weights)
        return self.tracknet_model
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

        # batch['img'] = batch['img'].to(self.device, non_blocking=True)

        # if self.args.half and self.device.type == "cuda":
        #     batch['img'] = batch['img'].half() / 255.0  # `float16`
        # else:
        #     batch['img'] = batch['img'].float() / 255.0  # `float32`

        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch

    def get_validator(self):
        # self.loss_names = 'pos_loss', 'mov_loss', 'conf_loss', 'hit_loss'
        self.loss_names = 'pos_loss', 'conf_loss'
        return TrackNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        self.add_callback("print_confusion_matrix", self.tracknet_model.print_confusion_matrix())
        self.add_callback("init_conf_confusion", self.tracknet_model.init_conf_confusion())
        return ('\n' + '%11s' *
                (3 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Size')
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLOv5 training."""
        pass
    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self._save_dataset_config()

    def _save_dataset_config(self):
        try:
            train_ds = self.train_loader.dataset
            val_ds   = self.test_loader.dataset

            train_cfg = train_ds.get_dataset_config() \
                        if hasattr(train_ds, 'get_dataset_config') else {}
            val_cfg   = val_ds.get_dataset_config() \
                        if hasattr(val_ds, 'get_dataset_config') else {}

            config = {
                "generated_at": datetime.now().isoformat(),
                "save_dir": str(self.save_dir),
                "train_dataset": train_cfg,
                "val_dataset":   val_cfg,
            }

            out_path = os.path.join(self.save_dir, "dataset_config.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"\n[INFO] Dataset config saved → {out_path}")

        except Exception as e:
            print(f"[WARN] Failed to save dataset config: {e}")