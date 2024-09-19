from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.tracknet_v4 import TrackNetV4
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from copy import copy


class TrackNetTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return TrackNetDataset(root_dir=img_path)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TrackNetV4(cfg, ch=10, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    def get_validator(self):
        # self.loss_names = 'pos_loss', 'mov_loss', 'conf_loss', 'hit_loss'
        self.loss_names = 'pos_loss', 'conf_loss', 'iou_loss'
        return TrackNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (3 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Size')
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLOv5 training."""
        pass
    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass