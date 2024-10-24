
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results

from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.torch_utils import select_device


class TrackNetPredictor(BasePredictor):
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = model
        self.device = select_device(self.args.device, verbose=verbose)  # update device
        self.args.half = self.model.fp16  # update half
        # self.args.half = self.args.half  # update half
        self.model.eval()
    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results