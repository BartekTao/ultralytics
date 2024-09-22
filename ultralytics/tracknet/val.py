import numpy as np
import torch
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.utils.transform import target_grid
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.metrics import DetMetrics
from sklearn.metrics import confusion_matrix

class TrackNetValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
    
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
        self.stride = 32
        self.num_groups = 10

        self.total_loss = 0.0
        self.num_samples = 0
        self.conf_TP = 0
        self.conf_TN = 0
        self.conf_FP = 0
        self.conf_FN = 0
        self.conf_acc = 0
        self.conf_precision = 0
        self.pos_TP = 0
        self.pos_TN = 0
        self.pos_FP = 0
        self.pos_FN = 0
        self.pos_acc = 0
        self.pos_precision = 0
        self.ball_count = 0
        self.pred_ball_count = 0
        device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            self.update_metrics_once(0, preds, batch_target[0])
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [330 * 20 * 20]
        # batch_target = [10*7]
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
        
        target_pos_distri = torch.zeros(self.num_groups, 20, 20, self.feat_no, device=self.device)
        mask_has_ball = torch.zeros(self.num_groups, 20, 20, device=self.device)
        cls_targets = torch.zeros(self.num_groups, 20, 20, 1, device=self.device)

        for target_idx, target in enumerate(batch_target):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], self.stride)
                mask_has_ball[target_idx, grid_y, grid_x] = 1
                
                target_pos_distri[target_idx, grid_y, grid_x, 0] = offset_x*(self.reg_max-1)/self.stride
                target_pos_distri[target_idx, grid_y, grid_x, 1] = offset_y*(self.reg_max-1)/self.stride

                ## cls
                cls_targets[target_idx, grid_y, grid_x, 0] = 1
        
        target_pos_distri = target_pos_distri.view(self.num_groups*20*20, self.feat_no)
        cls_targets = cls_targets.view(self.num_groups*20*20, 1)
        mask_has_ball = mask_has_ball.view(self.num_groups*20*20).bool()

        # 計算 conf 的 confusion matrix
        threshold = 0.8
        pred_binary = (pred_probs >= threshold)
        self.pred_ball_count += pred_binary.int().sum()

        unique_classes = torch.unique(cls_targets.bool())
        if len(unique_classes) == 1:
            if unique_classes.item() == 1:
                # All targets are 1 (positive class)
                self.conf_TP += (pred_binary == 1).sum().item()  # Count of true positives
                self.conf_FN += (pred_binary == 0).sum().item()  # Count of false negatives
                self.conf_TN += 0  # No true negatives
                self.conf_FP += 0  # No false positives
            else:
                # All targets are 0 (negative class)
                self.conf_TN += (pred_binary == 0).sum().item()  # Count of true negatives
                self.conf_FP += (pred_binary == 1).sum().item()  # Count of false positives
                self.conf_TP += 0  # No true positives
                self.conf_FN += 0  # No false negatives
        else:
            # Compute confusion matrix normally
            conf_matrix = confusion_matrix(cls_targets.bool(), pred_binary)
            self.conf_TN += conf_matrix[0][0]
            self.conf_FP += conf_matrix[0][1]
            self.conf_FN += conf_matrix[1][0]
            self.conf_TP += conf_matrix[1][1]

        # 計算 x, y 的 confusion matrix
        pred_tensor = pred_pos[mask_has_ball]
        ground_truth_tensor = target_pos_distri[mask_has_ball]
        ball_count = mask_has_ball.sum()
        self.ball_count += ball_count
        

        tolerance = 1
        x_tensor_correct = (torch.abs(pred_tensor[:, 0] - ground_truth_tensor[:, 0]) <= tolerance).int()
        y_tensor_correct = (torch.abs(pred_tensor[:, 1] - ground_truth_tensor[:, 1]) <= tolerance).int()

        tensor_combined_correct = (x_tensor_correct & y_tensor_correct).int()

        ground_truth_binary_tensor = torch.ones(ball_count).int()

        unique_classes = torch.unique(ground_truth_binary_tensor)
        if ball_count == 0:
            print("There are no balls.")
        elif len(unique_classes) == 1:
            if unique_classes.item() == 1:
                # All targets are 1 (positive class)
                self.pos_TP += (tensor_combined_correct == 1).sum().item()  # Count of true positives
                self.pos_FN += (tensor_combined_correct == 0).sum().item()  # Count of false negatives
                self.pos_TN += 0  # No true negatives
                self.pos_FP += 0  # No false positives
            else:
                # All targets are 0 (negative class)
                self.pos_TN += (tensor_combined_correct == 0).sum().item()  # Count of true negatives
                self.pos_FP += (tensor_combined_correct == 1).sum().item()  # Count of false positives
                self.pos_TP += 0  # No true positives
                self.pos_FN += 0  # No false negatives
        else:
            # Compute confusion matrix normally
            pos_matrix = confusion_matrix(ground_truth_binary_tensor, tensor_combined_correct)
            self.pos_TN += pos_matrix[0][0]
            self.pos_FP += pos_matrix[0][1]
            self.pos_FN += pos_matrix[1][0]
            self.pos_TP += pos_matrix[1][1]

        
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        if (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP) != 0:
            self.pos_acc = (self.pos_TN + self.pos_TP) / (self.pos_FN+self.pos_FP+self.pos_TN + self.pos_TP)
        if (self.conf_FN+self.conf_FP+self.conf_TN + self.conf_TP) != 0:
            self.conf_acc = (self.conf_TN + self.conf_TP) / (self.conf_FN+self.conf_FP+self.conf_TN + self.conf_TP)
        if (self.conf_TP+self.conf_FP) != 0:
            self.conf_precision = self.conf_TP/(self.conf_TP+self.conf_FP)
        if (self.pos_TP+self.pos_FP) != 0:
            self.pos_precision = self.pos_TP/(self.pos_TP+self.pos_FP)

    def get_stats(self):
        """Return the stats."""
        return {'pos_FN': self.pos_FN, 'pos_FP': self.pos_FP, 'pos_TN': self.pos_TN, 
                'pos_TP': self.pos_TP, 'pos_acc': self.pos_acc, 'pos_precision': self.pos_precision,
                'conf_FN': self.conf_FN, 'conf_FP': self.conf_FP, 'conf_TN': self.conf_TN, 
                'conf_TP': self.conf_TP, 'conf_acc': self.conf_acc, 'conf_precision': self.conf_precision,
                'threshold>0.8 rate':self.pred_ball_count/self.ball_count}
    
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
