import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.tracknet.utils.plotting import display_image_with_coordinates, display_predict_in_checkerboard
from ultralytics.tracknet.utils.transform import target_grid

from ultralytics.yolo.utils import LOGGER

# check_training_img_path = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img\img_'
# check_training_img_path = r'/usr/src/datasets/tracknet/visualize_train_img/img_'
# check_val_img_path = r'/usr/src/datasets/tracknet/visualize_val_img/img_'

class TrackNetLoss:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        self.hyp = h

        m = model.model[-1]  # Detect() module
        pos_weight = torch.tensor(self.hyp.weight_hit).to(device)
        self.hit_bce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
        self.mse = nn.MSELoss(reduction='sum')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.batch_count = 0
        self.train_count = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.sample_path = os.path.join(self.hyp.save_dir, "training_samples")

    def __call__(self, preds, batch):
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0].to(self.device) # only pick first (stride = 16)

        if preds.requires_grad:
            self.train_count += 1
        batch_target = batch['target'].to(self.device)
        batch_img = batch['img'].to(self.device)

        if self.hyp.use_dxdy_loss:
            print("use_dxdy_loss")
            loss = torch.zeros(4, device=self.device)
        else:
            print("!use_dxdy_loss")
            loss = torch.zeros(3, device=self.device)
            
        batch_size = preds.shape[0]

        # for each batch
        for idx, pred in enumerate(preds):
            # pred = [60 * 20 * 20]
            stride = self.stride[0]
            pred_distri, pred_scores, pred_hits = torch.split(pred, [40, 10, 10], dim=0)
            pred_distri = pred_distri.reshape(4, 10, 20, 20)
            pred_pos, pred_mov = torch.split(pred_distri, [2, 2], dim=0)

            pred_pos = pred_pos.permute(1, 0, 2, 3).contiguous()
            pred_mov = pred_mov.permute(1, 0, 2, 3).contiguous()

            pred_pos = torch.sigmoid(pred_pos)
            target_pos = pred_pos.detach().clone()
            pred_mov = torch.tanh(pred_mov)
            target_mov = pred_mov.detach().clone()
            
            cls_targets = torch.zeros(pred_scores.shape, device=self.device)
            hit_targets = torch.zeros(pred_hits.shape, device=self.device)
            
            position_loss = torch.tensor(0.0, device=self.device)
            move_loss = torch.tensor(0.0, device=self.device)
            
            mask_has_ball = torch.zeros_like(target_pos)
            for target_idx, target in enumerate(batch_target[idx]):
                if target[6] == 1:
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    hit_targets[target_idx, grid_y, grid_x] = 1
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    mask_has_ball[target_idx, :, grid_y, grid_x] = 1
                    
                    target_pos[target_idx, 0, grid_y, grid_x] = offset_x/stride
                    target_pos[target_idx, 1, grid_y, grid_x] = offset_y/stride

                    target_mov[target_idx, 0, grid_y, grid_x] = target[4]/640
                    target_mov[target_idx, 1, grid_y, grid_x] = target[5]/640

                    ## cls
                    cls_targets[target_idx, grid_y, grid_x] = 1
            position_loss = 32*self.mse(pred_pos, target_pos)

            if self.hyp.use_dxdy_loss:
                move_loss = 640*self.mse(pred_mov, target_mov) # / (1 if mask_has_ball.sum() == 0 else mask_has_ball.sum())

            conf_loss = tracknet_conf_loss(cls_targets, pred_scores, [1, self.hyp.weight_conf], self.batch_count)
            hit_loss = self.hit_bce(pred_hits, hit_targets)
            # conf_loss = focal_loss(pred_scores, cls_targets, alpha=[0.998, 0.002], weight=weight_conf)

            if torch.isnan(position_loss).any() or torch.isinf(position_loss).any():
                LOGGER.warning("NaN or Inf values in position_loss!")
            if torch.isnan(conf_loss).any() or torch.isinf(conf_loss).any():
                LOGGER.warning("NaN or Inf values in conf_loss!")

            if pred_scores.requires_grad:
                pred_binary = pred_scores >= 0.5
                self.TP += torch.sum((pred_binary == 1) & (cls_targets == 1))
                self.FP += torch.sum((pred_binary == 1) & (cls_targets == 0))
                self.TN += torch.sum((pred_binary == 0) & (cls_targets == 0))
                self.FN += torch.sum((pred_binary == 0) & (cls_targets == 1))
                

            # check
            # if (self.batch_count%400 == 0 and pred_scores.requires_grad and idx == 15) or (self.batch_count%20 == 0 and not pred_scores.requires_grad and idx == 15):
            #     pred_conf_all = torch.sigmoid(pred_scores.detach()).cpu()
            #     pred_hits_all = torch.sigmoid(pred_hits.detach()).cpu()
            #     pred_mov_all = pred_mov.detach().clone()
            #     pred_pos_all = pred_pos.detach().clone()
            #     for rand_idx in range(10):
            #         pred_conf = pred_conf_all[rand_idx]
            #         pred__hit = pred_hits_all[rand_idx]
            #         img = batch_img[idx][rand_idx]
            #         x = int(batch_target[idx][rand_idx][2].item())
            #         y = int(batch_target[idx][rand_idx][3].item())
            #         dx = int(batch_target[idx][rand_idx][4].item())
            #         dy = int(batch_target[idx][rand_idx][5].item())
            #         hit = int(batch_target[idx][rand_idx][6].item())

            #         pred_conf_np = pred_conf.numpy()
            #         y_positions, x_positions = np.where(pred_conf_np >= 0.5)
            #         pred_coordinates = list(zip(x_positions, y_positions))

            #         pred_list = []
            #         for pred_x_coordinates, pred_y_coordinates in pred_coordinates:
            #             pred_conf_format = "{:.2f}".format(pred_conf[pred_y_coordinates][pred_x_coordinates].item())
            #             pred_hit_format = "{:.2f}".format(pred__hit[pred_y_coordinates][pred_x_coordinates].item())
            #             pred_list.append((pred_x_coordinates, 
            #                               pred_y_coordinates, 
            #                               pred_pos_all[rand_idx][0][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_pos_all[rand_idx][1][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_mov_all[rand_idx][0][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_mov_all[rand_idx][1][pred_y_coordinates][pred_x_coordinates].item(), 
            #                               pred_conf_format,
            #                               pred_hit_format))

            #         filename = f'{self.batch_count//1185}_{int(self.batch_count%1185)}_{rand_idx}_{pred_scores.requires_grad}'

            #         count_ge_05 = np.count_nonzero(pred_conf >= 0.5)
            #         count_lt_05 = np.count_nonzero(pred_conf < 0.5)
            #         loss_dict = {}
            #         loss_dict['conf_loss'] = conf_loss.item()
            #         loss_dict['position_loss'] = position_loss.item()
            #         loss_dict['moving_loss'] = move_loss.item()
            #         loss_dict['pred_conf >= 0.5 count'] = count_ge_05
            #         loss_dict['pred_conf < 0.5 count'] = count_lt_05
            #         loss_dict['x, y'] = (x%32, y%32)
            #         loss_dict['pred_x, pred_y'] = (pred_pos_all[rand_idx][0][int(y//32)][int(x//32)].item()*32, pred_pos_all[rand_idx][1][int(y//32)][int(x//32)].item()*32)
            #         loss_dict['dx, dy'] = (dx, dy)
            #         loss_dict['pred_dx, pred_dy'] = (pred_mov_all[rand_idx][0][int(y//32)][int(x//32)].item()*640, pred_mov_all[rand_idx][1][int(y//32)][int(x//32)].item()*640)

            #         display_predict_in_checkerboard([(x, y, dx, dy, hit)], pred_list, 'board_'+filename, loss_dict)
            #         display_image_with_coordinates(img, [(x, y, dx, dy)], pred_list, filename, loss_dict)

            if self.hyp.use_dxdy_loss:
                loss[0] += position_loss * self.hyp.weight_pos
                loss[1] += move_loss * self.hyp.weight_mov
                loss[2] += conf_loss
                loss[3] += hit_loss
            else:
                loss[0] += position_loss * self.hyp.weight_pos
                loss[1] += conf_loss
                loss[2] += hit_loss

        tlose = loss.sum() * batch_size
        tlose_item = loss.detach()
        self.batch_count+=1

        if preds.requires_grad and self.train_count >= 1185 and self.train_count%1185 == 0:
            if self.TP > 0:
                precision = self.TP/(self.TP+self.FP)
                recall = self.TP/(self.TP+self.FN)
                f1 = (2*precision*recall)/(precision+recall)
            acc = (self.TN + self.TP) / (self.FN+self.FP+self.TN + self.TP)
            print(f"\nTraining Accuracy: {acc:.4f}, Training Precision: {precision:.4f}, Training Recall: {recall:.4f}, , Training F1-Score: {f1:.4f}\n")
            self.TP = 0
            self.FP = 0
            self.TN = 0
            self.FN = 0
        return tlose, tlose_item
def save_pred_and_loss(predictions, loss, filename, t_xy):
    """
    Save the predictions and loss into a CSV file.
    :param predictions: A tensor of shape [10, 20, 20] containing the predictions with gradient information.
    :param loss: A scalar representing the loss value.
    :param filename: The name of the file to save the data.
    """
    # Detach predictions from the current graph and convert to numpy
    if predictions.requires_grad:
        predictions = predictions.detach()
        loss = loss.detach()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    with open(filename+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each 20x20 prediction along with the loss
        i=0
        for pred in predictions:
            max_position = torch.argmax(t_xy[i])
            max_y, max_x = np.unravel_index(max_position, t_xy[i].shape)
            # Flatten the 20x20 prediction to a single row
            flattened_pred = pred.flatten()
            # Append the loss and write to the file
            writer.writerow(list(flattened_pred) + [loss] +[(max_x, max_y)])
            i+=1
def save_pos_mov_loss(predictions, loss, filename, target):
    """
    Save the predictions and loss into a CSV file.

    :param predictions: A tensor of shape [10, 20, 20] containing the predictions with gradient information.
    :param loss: A scalar representing the loss value.
    :param filename: The name of the file to save the data.
    """
    # Detach predictions from the current graph and convert to numpy
    if predictions.requires_grad:
        predictions = predictions.detach()
        loss = loss.detach()

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    with open(filename+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write each 20x20 prediction along with the loss
        i=0
        for pred in predictions:
            # Flatten the 20x20 prediction to a single row
            flattened_pred = pred.flatten()
            # Append the loss and write to the file
            writer.writerow(list(flattened_pred) + [loss])
            i+=1
        for pred in target:
            # Flatten the 20x20 prediction to a single row
            flattened_pred = pred.flatten()
            # Append the loss and write to the file
            writer.writerow(list(flattened_pred))
            i+=1
            


def tracknet_conf_loss(y_true, y_pred, class_weight, batch_count):
    y_pred = torch.sigmoid(y_pred)  

    custom_weights = torch.square(1 - y_pred) * y_true + torch.square(y_pred) * (1 - y_true)
    # custom_weights = (1 - y_pred) * y_true + (y_pred) * (1 - y_true)

    # y_true_squeeze  = y_true.any(dim=0, keepdim=True).repeat(10, 1, 1).int()

    class_weights = class_weight[0] * (1 - y_true) + class_weight[1] * y_true

    loss = (-1) * class_weights * custom_weights * (y_true * torch.log(torch.clamp(y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)) + 
                                                    (1 - y_true) * torch.log(torch.clamp(1 - y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)))
    
    loss = torch.sum(loss)
    
    # save loss detail
    # if (batch_count%400 == 0 and y_pred.requires_grad) or (batch_count%20 == 0 and not y_pred.requires_grad):
    #     filename = f'{self.sample_path}/{batch_count//1185}_{int(batch_count%1185)}_{y_pred.requires_grad}'
    #     y_true_cpu = y_true.cpu()
    #     save_pred_and_loss(y_pred, loss, filename, y_true_cpu)
    return loss

def focal_loss(pred_logits, targets, alpha=0.95, gamma=2.0, epsilon=1e-3, weight=10):
    """
    :param pred_logits: 預測的logits, shape [batch_size, 1, H, W]
    :param targets: 真實標籤, shape [batch_size, 1, H, W]
    :param alpha: 用於平衡正、負樣本的權重。這裡可以是一個scalar或一個list[alpha_neg, alpha_pos]。
    :param gamma: 用於調節著重於正確或錯誤預測的程度
    :return: focal loss
    """
    pred_probs = torch.sigmoid(pred_logits)

    pred_probs = torch.clamp(pred_probs, epsilon, 1.0-epsilon)  # log(0) 會導致無窮大
    if isinstance(alpha, (list, tuple)):
        alpha_neg = alpha[0]
        alpha_pos = alpha[1]
    else:
        alpha_neg = (1 - alpha)
        alpha_pos = alpha

    pt = torch.where(targets == 1, pred_probs, 1 - pred_probs)
    alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
    
    ce_loss = -torch.log(pt)
    # if torch.isinf(ce_loss).any():
    #     LOGGER.warning("ce_loss value is infinite!")
    fl = alpha_t * (1 - pt) ** gamma * ce_loss
    test = fl.mean() * weight_conf
    return test