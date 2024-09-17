import torch
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.utils.transform import target_grid
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.metrics import DetMetrics

class TrackNetValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess(self, batch):
        """In this case, the preprocessing step is mainly handled by the dataloader."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.total_loss = 0.0
        self.num_samples = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
        self.hasMax = 0
        self.hasBall = 0
        device = next(model.parameters()).device
        self.reg_max = 16
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.no = 33
        self.feat_no = 2
        self.nc = 1
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_size = preds.shape[0]
        if preds.shape == (330, 20, 20):
            self.update_metrics_once(0, preds, batch_target)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [330 * 20 * 20]
        # batch_target = [10*6]
        feats = pred
        pred_distri, pred_scores = feats.view(self.no, -1).split(
            (self.reg_max * self.feat_no, self.nc), 0)
        
        pred_scores = pred_scores.permute(1, 0).contiguous()
        pred_distri = pred_distri.permute(1, 0).contiguous()

        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*20*20]
        
        a, c = pred_distri.shape

        pred_pos = pred_distri.view(a, self.feat_no, c // self.feat_no).softmax(2).matmul(
            self.proj.type(pred_distri.dtype))

        # max_values_dim1, max_indices_dim1 = pred_probs.max(dim=2)
        # final_max_values, max_indices_dim2 = max_values_dim1.max(dim=1)
        # max_positions = [(index.item(), max_indices_dim1[i, index].item()) for i, index in enumerate(max_indices_dim2)]

        # #targets = pred_distri.clone().detach()
        # #cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2])
        # stride = 32
        # if len(batch_target.shape) == 3:
        #     batch_target = batch_target[0]
        # for idx, target in enumerate(batch_target):
        #     if target[1] == 1:
        #         # xy
        #         grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
        #         if (grid_x > 20 or grid_y > 20):
        #             LOGGER.Warning("target grid transform error")
        #         if (pred_probs[idx][grid_x][grid_y] > 0.5):
        #             self.hasBall += 1
                
        #         # print(f"target: {(grid_x, grid_y, offset_x, offset_y)}, ")
        #         # print(f"predict_conf: {pred_probs[idx][grid_x][grid_y]}, ")
        #         # print(f"pred_pos: {pred_pos[idx][grid_x][grid_y]}")
        #         # print(pred_probs[idx][max_positions[idx]])
        #         # print(max_positions[idx])
        #         if pred_probs[idx][max_positions[idx]] > 0.5:
        #             self.hasMax += 1
        #             x, y = max_positions[idx]
        #             real_x = x*stride + pred_pos[idx][x][y] #*stride
        #             real_y = y*stride + pred_pos[idx][x][y] #*stride
        #             if (grid_x, grid_y) == max_positions[idx]:
        #                 self.TP+=1
        #             else:
        #                 self.FN+=1
        #         else:
        #             self.FN+=1
        #     elif pred_probs[idx][max_positions[idx]] > 0.5:
        #         self.FP+=1
        #     else:
        #         self.TN+=1
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        # self.acc = (self.TN + self.TP) / (self.FN+self.FP+self.TN + self.TP)

    def get_stats(self):
        """Return the stats."""
        return {'FN': self.FN, 'FP': self.FP, 'TN': self.TN, 'TP': self.TP, 'acc': self.acc, 'max_conf>0.5': self.hasMax, 'correct_cell>0.5':self.hasBall}
    
    def print_results(self):
        """Print the results."""
        # precision = 0
        # recall = 0
        # f1 = 0
        # if self.TP > 0:
        #     precision = self.TP/(self.TP+self.FP)
        #     recall = self.TP/(self.TP+self.FN)
        #     f1 = (2*precision*recall)/(precision+recall)
        # print(f"Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"


class TrackNetValidatorWithHit(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess(self, batch):
        """In this case, the preprocessing step is mainly handled by the dataloader."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.total_loss = 0.0
        self.num_samples = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
        self.hasMax = 0
        self.hasBall = 0
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_size = preds.shape[0]
        if preds.shape == (60, 20, 20):
            self.update_metrics_once(0, preds, batch_target)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [50 * 20 * 20]
        # batch_target = [10*6]
        pred_distri, pred_scores, pred_hits = torch.split(pred, [40, 10, 10], dim=0)
        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*20*20]
        
        pred_pos, pred_mov = torch.split(pred_distri, [20, 20], dim=0)
        # pred_pos = torch.sigmoid(pred_pos)
        # pred_mov = torch.tanh(pred_mov)

        max_values_dim1, max_indices_dim1 = pred_probs.max(dim=2)
        final_max_values, max_indices_dim2 = max_values_dim1.max(dim=1)
        max_positions = [(index.item(), max_indices_dim1[i, index].item()) for i, index in enumerate(max_indices_dim2)]

        #targets = pred_distri.clone().detach()
        #cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2])
        stride = 32
        if len(batch_target.shape) == 3:
            batch_target = batch_target[0]
        for idx, target in enumerate(batch_target):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                if (grid_x > 20 or grid_y > 20):
                    LOGGER.Warning("target grid transform error")
                if (pred_probs[idx][grid_x][grid_y] > 0.5):
                    self.hasBall += 1
                
                # print(f"target: {(grid_x, grid_y, offset_x, offset_y)}, ")
                # print(f"predict_conf: {pred_probs[idx][grid_x][grid_y]}, ")
                # print(f"pred_pos: {pred_pos[idx][grid_x][grid_y]}")
                # print(pred_probs[idx][max_positions[idx]])
                # print(max_positions[idx])
                if pred_probs[idx][max_positions[idx]] > 0.5:
                    self.hasMax += 1
                    x, y = max_positions[idx]
                    real_x = x*stride + pred_pos[idx][x][y] #*stride
                    real_y = y*stride + pred_pos[idx][x][y] #*stride
                    if (grid_x, grid_y) == max_positions[idx]:
                        self.TP+=1
                    else:
                        self.FN+=1
                else:
                    self.FN+=1
            elif pred_probs[idx][max_positions[idx]] > 0.5:
                self.FP+=1
            else:
                self.TN+=1
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        self.acc = (self.TN + self.TP) / (self.FN+self.FP+self.TN + self.TP)

    def get_stats(self):
        """Return the stats."""
        return {'FN': self.FN, 'FP': self.FP, 'TN': self.TN, 'TP': self.TP, 'acc': self.acc, 'max_conf>0.5': self.hasMax, 'correct_cell>0.5':self.hasBall}
    
    def print_results(self):
        """Print the results."""
        precision = 0
        recall = 0
        f1 = 0
        if self.TP > 0:
            precision = self.TP/(self.TP+self.FP)
            recall = self.TP/(self.TP+self.FN)
            f1 = (2*precision*recall)/(precision+recall)
        print(f"Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"
