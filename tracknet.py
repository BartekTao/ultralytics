
import argparse
from copy import copy
import csv
import os
import pathlib
import time
from matplotlib import patheffects
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.engine.model import TrackNet
from ultralytics.tracknet.predict import TrackNetPredictor
from ultralytics.tracknet.video_dataset import TrackNetVideoDataset
from ultralytics.tracknet.test_dataset import TrackNetTestDataset
from ultralytics.tracknet.train import TrackNetTrainer
from ultralytics.tracknet.utils.confusion_matrix import ConfConfusionMatrix
from ultralytics.tracknet.utils.loss import TrackNetLoss
from ultralytics.tracknet.utils.plotting import display_image_with_coordinates, display_predict_image
from ultralytics.tracknet.utils.transform import target_grid
from ultralytics.tracknet.val_dataset import TrackNetValDataset
from ultralytics.yolo.data import dataloaders
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.yolo.utils import LOGGER, RANK, TQDM_BAR_FORMAT, ops
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, callbacks,
                                    is_git_dir, yaml_load)
from ultralytics.yolo.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import gc

# from ultralytics import YOLO

# # Create a new YOLO model from scratch
# model = YOLO(r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)
# # results = model.train(data='tracknet.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

def parse_video_path(video_path, save_dir):
    path = Path(video_path)
    video_name = path.stem
    
    if path.parent.name == 'video':
        # 結構：.../parent_dir/video/name.mp4
        parent_dir = path.parent.parent.name
        datasets_base = path.parent.parent
    else:
        # 結構：.../parent_dir/name.mp4（影片直接在父資料夾下）
        parent_dir = path.parent.name
        datasets_base = path.parent
    
    # 如果 parent_dir 為空或是根目錄，使用 "unknown"
    if not parent_dir or parent_dir in ['/', '.', '..']:
        parent_dir = "unknown"
        LOGGER.warning(f"Could not determine parent directory from path: {video_path}")
        LOGGER.warning(f"Using parent_dir='unknown'")
    
    return {
        'video_name': video_name,
        'parent_dir': parent_dir,
        'datasets_base': datasets_base,
        # 輸出路徑
        'output_video': os.path.join(save_dir, parent_dir, 'video', f'{video_name}_all_points.mp4'),
        'output_csv': os.path.join(save_dir, parent_dir, 'csv', video_name, 'all.csv'),
        'output_frame_runs': os.path.join(save_dir, parent_dir, 'frame', video_name),
        'output_frame_datasets': datasets_base / 'frame' / video_name
    }

def main(arg):
    overrides = {}
    overrides['model'] = arg.model_path
    overrides['mode'] = arg.mode
    overrides['data'] = arg.data
    overrides['epochs'] = arg.epochs
    overrides['plots'] = arg.plots
    overrides['batch'] = arg.batch
    overrides['patience'] = 300
    overrides['mosaic'] = 0.0
    overrides['plots'] = arg.plots
    overrides['val'] = arg.val
    overrides['use_dxdy_loss'] = arg.use_dxdy_loss
    overrides['use_resampler'] = arg.use_resampler
    overrides['save_period'] = 10
    overrides['workers'] = 16
    overrides['device'] = 0
    overrides['background_method'] = arg.background_method

    if arg.mode == 'train':
        trainer = TrackNetTrainer(overrides=overrides)
        # trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
        trainer.train()
    elif arg.mode == 'predict':
        model, _ = attempt_load_one_weight(arg.model_path)
        worker = 0
        if torch.cuda.is_available():
            model.cuda()
            worker = 1
        dataset = TrackNetTestDataset(root_dir=arg.source)
        dataloader = build_dataloader(dataset, arg.batch, worker, shuffle=False, rank=-1)
        overrides = overrides.copy()
        overrides['save'] = False
        predictor = TrackNetPredictor(overrides=overrides)
        predictor.setup_model(model=model, verbose=False)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), bar_format=TQDM_BAR_FORMAT)
        elapsed_times = 0.0

        metrics = []
        for i, batch in pbar:
            input_data = batch['img']
            idx = np.random.randint(0, 10)
            idx = 5
            # hasBall = target[idx][1].item()
            # t_x = target[idx][2].item()
            # t_y = target[idx][3].item()
            # xy = [(t_x, t_y)]
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            start_time = time.time()

            # [1*1*60*20*20]
            # [6*20*20]
            pred = predictor.inference(input_data)[0][0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            elapsed_times+=elapsed_time
            pbar.set_description(f'{elapsed_times / (i+1):.2f}  {i+1}/{len(pbar)}')
            
            feats = pred
            pred_distri, pred_scores = feats.view(33, -1).split(
                (16 * 2, 1), 0)
            
            pred_scores = pred_scores.permute(1, 0).contiguous()
            pred_distri = pred_distri.permute(1, 0).contiguous()

            pred_probs = torch.sigmoid(pred_scores)
            a, c = pred_distri.shape

            device = next(model.parameters()).device
            proj = torch.arange(16, dtype=torch.float, device=device)
            pred_pos = pred_distri.view(a, 2, c // 2).softmax(2).matmul(
                proj.type(pred_distri.dtype))

            pred_scores = pred_probs.view(10, 20, 20)
            pred_pos_x, pred_pos_y = pred_pos.view(10, 20, 20, 2).split([1, 1], dim=3)

            pred_pos_x = pred_pos_x.squeeze(-1)
            pred_pos_y = pred_pos_y.squeeze(-1)

            ####### 檢視 conf 訓練狀況 #######
            # ms = []
            # p_conf = pred_scores[0]
            # p_cell_x = pred_pos_x[0]
            # p_cell_y = pred_pos_y[0]

            # greater_than_05_positions = torch.nonzero(p_conf > 0.8, as_tuple=False)
            
            # for position in greater_than_05_positions:
            #     t_p_conf = p_conf[position[0], position[1]]
                
            #     # position 位置是否需要調換
            #     t_p_cell_x = p_cell_x[position[0], position[1]]
            #     t_p_cell_y = p_cell_y[position[0], position[1]]

            #     metric = {}
            #     metric["grid_x"] = position[1]
            #     metric["grid_y"] = position[0]
            #     metric["x"] = t_p_cell_x/15
            #     metric["y"] = t_p_cell_y/15
            #     metric["conf"] = t_p_conf

            #     ms.append(metric)

            # display_predict_image(
            #         input_data[0][0],  
            #         ms, 
            #         str(i),
            #         )
            
            ####### 檢視 max conf #######
            for frame_idx in range(10):
                p_conf = pred_scores[frame_idx]
                p_cell_x = pred_pos_x[frame_idx]
                p_cell_y = pred_pos_y[frame_idx]

                max_position = torch.argmax(p_conf)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x]

                metric = {}
                metric["grid_x"] = max_x
                metric["grid_y"] = max_y
                metric["x"] = p_cell_x[max_y][max_x]/16
                metric["y"] = p_cell_y[max_y][max_x]/16
                metric["conf"] = max_conf

                if i == 0 or frame_idx == 9:
                    metrics.append(metric)
                elif i > 0 and metric["conf"] > metrics[i+frame_idx]["conf"]:
                    metrics[i+frame_idx] = metric

            first_metric = metrics[i]
            display_predict_image(
                    input_data[0][0],  
                    [first_metric], 
                    str(i),
                    )

        print(f"avg predict time: { elapsed_times / len(dataloader):.2f} 毫秒")
    elif arg.mode == 'val':
        conf_TP = 0
        conf_TN = 0
        conf_FP = 0
        conf_FN = 0
        conf_acc = 0
        conf_precision = 0
        pos_TP = 0
        pos_TN = 0
        pos_FP = 0
        pos_FN = 0
        pos_acc = 0
        pos_precision = 0
        ball_count = 0
        pred_ball_count = 0
        stride = 32

        model, _ = attempt_load_one_weight(arg.model_path)
        model.eval()
        worker = 0
        if torch.cuda.is_available():
            model.cuda()
            worker = 1
        dataset = TrackNetValDataset(root_dir=arg.source)
        dataloader = build_dataloader(dataset, arg.batch, worker, shuffle=False, rank=-1)
        overrides = overrides.copy()
        overrides['save'] = False
        predictor = TrackNetPredictor(overrides=overrides)
        predictor.setup_model(model=model, verbose=False)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), bar_format=TQDM_BAR_FORMAT)
        elapsed_times = 0.0

        metrics = []
        for i, batch in pbar:
            batch_target = batch['target'][0]
            input_data = batch['img']
            idx = np.random.randint(0, 10)
            idx = 5
            # hasBall = target[idx][1].item()
            # t_x = target[idx][2].item()
            # t_y = target[idx][3].item()
            # xy = [(t_x, t_y)]
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            start_time = time.time()

            # [1*1*60*20*20]
            # [6*20*20]
            pred = predictor.inference(input_data)[0][0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            elapsed_times+=elapsed_time
            pbar.set_description(f'{elapsed_times / (i+1):.2f}  {i+1}/{len(pbar)}')
            
            feats = pred
            pred_distri, pred_scores, pred_dxdy = feats.view(35, -1).split(
                (16 * 2, 1, 2), 0)
            
            pred_scores = pred_scores.permute(1, 0).contiguous()
            pred_distri = pred_distri.permute(1, 0).contiguous()

            pred_dxdy = pred_dxdy.permute(1, 0).contiguous()
            pred_dxdy = torch.tanh(pred_dxdy)

            pred_probs = torch.sigmoid(pred_scores)
            a, c = pred_distri.shape

            device = next(model.parameters()).device
            proj = torch.arange(16, dtype=torch.float, device=device)
            pred_pos = pred_distri.view(a, 2, c // 2).softmax(2).matmul(
                proj.type(pred_distri.dtype))
            
            # set target
            target_pos_distri = torch.zeros(10, 20, 20, 2, device=device)
            mask_has_ball = torch.zeros(10, 20, 20, device=device)
            cls_targets = torch.zeros(10, 20, 20, 1, device=device)

            for target_idx, target in enumerate(batch_target):
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    mask_has_ball[target_idx, grid_y, grid_x] = 1
                    
                    target_pos_distri[target_idx, grid_y, grid_x, 0] = offset_x*(16-1)/stride
                    target_pos_distri[target_idx, grid_y, grid_x, 1] = offset_y*(16-1)/stride

                    ## cls
                    cls_targets[target_idx, grid_y, grid_x, 0] = 1
            target_pos_distri = target_pos_distri.view(10*20*20, 2)
            cls_targets = cls_targets.view(10*20*20, 1)
            mask_has_ball = mask_has_ball.view(10*20*20).bool()

            # 計算 conf 的 confusion matrix
            conf_matrix = ConfConfusionMatrix()
            conf_matrix.confusion_matrix(pred_probs, cls_targets)
            conf_TP += conf_matrix.conf_TP
            conf_FN += conf_matrix.conf_FN
            conf_TN += conf_matrix.conf_TN
            conf_FP += conf_matrix.conf_FP

            conf_matrix.print_confusion_matrix()

            # 計算 x, y 的 confusion matrix
            pred_tensor = pred_pos[mask_has_ball]
            ground_truth_tensor = target_pos_distri[mask_has_ball]
            ball_count = mask_has_ball.sum()
            print(f"ball_count: {ball_count}\n")
            ball_count += ball_count
            

            tolerance = 2
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
                    pos_TP += (tensor_combined_correct == 1).sum().item()  # Count of true positives
                    pos_FN += (tensor_combined_correct == 0).sum().item()  # Count of false negatives
                    pos_TN += 0  # No true negatives
                    pos_FP += 0  # No false positives
                else:
                    # All targets are 0 (negative class)
                    pos_TN += (tensor_combined_correct == 0).sum().item()  # Count of true negatives
                    pos_FP += (tensor_combined_correct == 1).sum().item()  # Count of false positives
                    pos_TP += 0  # No true positives
                    pos_FN += 0  # No false negatives
            else:
                # Compute confusion matrix normally
                pos_matrix = confusion_matrix_gpu(ground_truth_binary_tensor, tensor_combined_correct)
                pos_TN += pos_matrix[0][0]
                pos_FP += pos_matrix[0][1]
                pos_FN += pos_matrix[1][0]
                pos_TP += pos_matrix[1][1]

            pred_scores = pred_probs.view(10, 20, 20)
            pred_pos_x, pred_pos_y = pred_pos.view(10, 20, 20, 2).split([1, 1], dim=3)

            pred_pos_x = pred_pos_x.squeeze(-1)
            pred_pos_y = pred_pos_y.squeeze(-1)

            ####### 檢視 conf 訓練狀況 #######
            # ms = []
            # p_conf = pred_scores[0]
            # p_cell_x = pred_pos_x[0]
            # p_cell_y = pred_pos_y[0]

            # greater_than_05_positions = torch.nonzero(p_conf > 0.8, as_tuple=False)
            
            # for position in greater_than_05_positions:
            #     t_p_conf = p_conf[position[0], position[1]]
                
            #     # position 位置是否需要調換
            #     t_p_cell_x = p_cell_x[position[0], position[1]]
            #     t_p_cell_y = p_cell_y[position[0], position[1]]

            #     metric = {}
            #     metric["grid_x"] = position[1]
            #     metric["grid_y"] = position[0]
            #     metric["x"] = t_p_cell_x/15
            #     metric["y"] = t_p_cell_y/15
            #     metric["conf"] = t_p_conf

            #     ms.append(metric)

            # display_predict_image(
            #         input_data[0][frame_idx],  
            #         ms, 
            #         str(i*10+frame_idx),
            #         )
            
            ####### 檢視 max conf #######
            for frame_idx in range(10):
                p_conf = pred_scores[frame_idx]
                p_cell_x = pred_pos_x[frame_idx]
                p_cell_y = pred_pos_y[frame_idx]

                max_position = torch.argmax(p_conf)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x]

                metric = {}
                metric["grid_x"] = max_x
                metric["grid_y"] = max_y
                metric["x"] = p_cell_x[max_y][max_x]/16
                metric["y"] = p_cell_y[max_y][max_x]/16
                metric["conf"] = max_conf

                if i == 0 or frame_idx == 9:
                    metrics.append(metric)
                elif i > 0 and metric["conf"] > metrics[i+frame_idx]["conf"]:
                    metrics[i+frame_idx] = metric

                display_predict_image(
                        input_data[0][frame_idx],  
                        [metric], 
                        str(i*10+frame_idx),
                        )
        
        if (pos_FN+pos_FP+pos_TN + pos_TP) != 0:
            pos_acc = (pos_TN + pos_TP) / (pos_FN+pos_FP+pos_TN + pos_TP)
        if (conf_FN+conf_FP+conf_TN + conf_TP) != 0:
            conf_acc = (conf_TN + conf_TP) / (conf_FN+conf_FP+conf_TN + conf_TP)
        if (conf_TP+conf_FP) != 0:
            conf_precision = conf_TP/(conf_TP+conf_FP)
        if (pos_TP+pos_FP) != 0:
            pos_precision = pos_TP/(pos_TP+pos_FP)
        matrix = {'pos_FN': pos_FN, 'pos_FP': pos_FP, 'pos_TN': pos_TN, 
                'pos_TP': pos_TP, 'pos_acc': pos_acc, 'pos_precision': pos_precision,
                'conf_FN': conf_FN, 'conf_FP': conf_FP, 'conf_TN': conf_TN, 
                'conf_TP': conf_TP, 'conf_acc': conf_acc, 'conf_precision': conf_precision,
                'threshold>0.8 rate':pred_ball_count/ball_count}
        print(matrix)
        # val_confusion_matrix = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\val_confusion_matrix\matrix.csv'
        val_confusion_matrix = r'/usr/src/datasets/tracknet/val_confusion_matrix/matrix.csv'
        csv_path = pathlib.Path(val_confusion_matrix)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode='w', newline='') as file:
            fieldnames = matrix.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(matrix)

        print(f"avg predict time: { elapsed_times / len(dataloader):.2f} 毫秒")
    elif arg.mode == 'val_v1':
        conf_TP = 0
        conf_TN = 0
        conf_FP = 0
        conf_FN = 0
        conf_acc = 0
        conf_precision = 0
        pos_TP = 0
        pos_TN = 0
        pos_FP = 0
        pos_FN = 0
        pos_acc = 0
        pos_precision = 0
        ball_count = 0
        pred_ball_count = 0
        stride = 32

        model, _ = attempt_load_one_weight(arg.model_path)
        model.eval()
        worker = 0
        if torch.cuda.is_available():
            model.cuda()
            worker = 1
        dataset = TrackNetValDataset(root_dir=arg.source)
        dataloader = build_dataloader(dataset, arg.batch, worker, shuffle=False, rank=-1)
        overrides = overrides.copy()
        overrides['save'] = False
        predictor = TrackNetPredictor(overrides=overrides)
        predictor.setup_model(model=model, verbose=False)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), bar_format=TQDM_BAR_FORMAT)
        elapsed_times = 0.0

        metrics = []
        for i, batch in pbar:
            batch_target = batch['target'][0]
            input_data = batch['img']
            idx = np.random.randint(0, 10)
            idx = 5
            # hasBall = target[idx][1].item()
            # t_x = target[idx][2].item()
            # t_y = target[idx][3].item()
            # xy = [(t_x, t_y)]
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            start_time = time.time()

            # [1*1*60*20*20]
            # [6*20*20]
            pred = predictor.inference(input_data)[0][0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            elapsed_times+=elapsed_time
            pbar.set_description(f'{elapsed_times / (i+1):.2f}  {i+1}/{len(pbar)}')
            
            feats = pred
            pred_distri, pred_scores = feats.view(33, -1).split(
                (16 * 2, 1), 0)
            
            pred_scores = pred_scores.permute(1, 0).contiguous()
            pred_distri = pred_distri.permute(1, 0).contiguous()

            pred_probs = torch.sigmoid(pred_scores)
            a, c = pred_distri.shape

            device = next(model.parameters()).device
            proj = torch.arange(16, dtype=torch.float, device=device)
            pred_pos = pred_distri.view(a, 2, c // 2).softmax(2).matmul(
                proj.type(pred_distri.dtype))
            
            # set target
            target_pos_distri = torch.zeros(10, 20, 20, 2, device=device)
            mask_has_ball = torch.zeros(10, 20, 20, device=device)
            cls_targets = torch.zeros(10, 20, 20, 1, device=device)

            for target_idx, target in enumerate(batch_target):
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    mask_has_ball[target_idx, grid_y, grid_x] = 1
                    
                    target_pos_distri[target_idx, grid_y, grid_x, 0] = offset_x*(16-1)/stride
                    target_pos_distri[target_idx, grid_y, grid_x, 1] = offset_y*(16-1)/stride

                    ## cls
                    cls_targets[target_idx, grid_y, grid_x, 0] = 1
            target_pos_distri = target_pos_distri.view(10*20*20, 2)
            cls_targets = cls_targets.view(10*20*20, 1)
            mask_has_ball = mask_has_ball.view(10*20*20).bool()

            # 計算 conf 的 confusion matrix
            conf_matrix = ConfConfusionMatrix()
            conf_matrix.confusion_matrix(pred_probs, cls_targets)
            conf_TP += conf_matrix.conf_TP
            conf_FN += conf_matrix.conf_FN
            conf_TN += conf_matrix.conf_TN
            conf_FP += conf_matrix.conf_FP

            conf_matrix.print_confusion_matrix()

            # 計算 x, y 的 confusion matrix
            pred_tensor = pred_pos[mask_has_ball]
            ground_truth_tensor = target_pos_distri[mask_has_ball]
            ball_count = mask_has_ball.sum()
            print(f"ball_count: {ball_count}\n")
            ball_count += ball_count
            

            tolerance = 2
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
                    pos_TP += (tensor_combined_correct == 1).sum().item()  # Count of true positives
                    pos_FN += (tensor_combined_correct == 0).sum().item()  # Count of false negatives
                    pos_TN += 0  # No true negatives
                    pos_FP += 0  # No false positives
                else:
                    # All targets are 0 (negative class)
                    pos_TN += (tensor_combined_correct == 0).sum().item()  # Count of true negatives
                    pos_FP += (tensor_combined_correct == 1).sum().item()  # Count of false positives
                    pos_TP += 0  # No true positives
                    pos_FN += 0  # No false negatives
            else:
                # Compute confusion matrix normally
                pos_matrix = confusion_matrix_gpu(ground_truth_binary_tensor, tensor_combined_correct)
                pos_TN += pos_matrix[0][0]
                pos_FP += pos_matrix[0][1]
                pos_FN += pos_matrix[1][0]
                pos_TP += pos_matrix[1][1]

            pred_scores = pred_probs.view(10, 20, 20)
            pred_pos_x, pred_pos_y = pred_pos.view(10, 20, 20, 2).split([1, 1], dim=3)

            pred_pos_x = pred_pos_x.squeeze(-1)
            pred_pos_y = pred_pos_y.squeeze(-1)

            ####### 檢視 conf 訓練狀況 #######
            # ms = []
            # p_conf = pred_scores[0]
            # p_cell_x = pred_pos_x[0]
            # p_cell_y = pred_pos_y[0]

            # greater_than_05_positions = torch.nonzero(p_conf > 0.8, as_tuple=False)
            
            # for position in greater_than_05_positions:
            #     t_p_conf = p_conf[position[0], position[1]]
                
            #     # position 位置是否需要調換
            #     t_p_cell_x = p_cell_x[position[0], position[1]]
            #     t_p_cell_y = p_cell_y[position[0], position[1]]

            #     metric = {}
            #     metric["grid_x"] = position[1]
            #     metric["grid_y"] = position[0]
            #     metric["x"] = t_p_cell_x/15
            #     metric["y"] = t_p_cell_y/15
            #     metric["conf"] = t_p_conf

            #     ms.append(metric)

            # display_predict_image(
            #         input_data[0][frame_idx],  
            #         ms, 
            #         str(i*10+frame_idx),
            #         )
            
            ####### 檢視 max conf #######
            for frame_idx in range(10):
                p_conf = pred_scores[frame_idx]
                p_cell_x = pred_pos_x[frame_idx]
                p_cell_y = pred_pos_y[frame_idx]

                max_position = torch.argmax(p_conf)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x]

                metric = {}
                metric["grid_x"] = max_x
                metric["grid_y"] = max_y
                metric["x"] = p_cell_x[max_y][max_x]/16
                metric["y"] = p_cell_y[max_y][max_x]/16
                metric["conf"] = max_conf

                if i == 0 or frame_idx == 9:
                    metrics.append(metric)
                elif i > 0 and metric["conf"] > metrics[i+frame_idx]["conf"]:
                    metrics[i+frame_idx] = metric

                display_predict_image(
                        input_data[0][frame_idx],  
                        [metric], 
                        str(i*10+frame_idx),
                        )
        
        if (pos_FN+pos_FP+pos_TN + pos_TP) != 0:
            pos_acc = (pos_TN + pos_TP) / (pos_FN+pos_FP+pos_TN + pos_TP)
        if (conf_FN+conf_FP+conf_TN + conf_TP) != 0:
            conf_acc = (conf_TN + conf_TP) / (conf_FN+conf_FP+conf_TN + conf_TP)
        if (conf_TP+conf_FP) != 0:
            conf_precision = conf_TP/(conf_TP+conf_FP)
        if (pos_TP+pos_FP) != 0:
            pos_precision = pos_TP/(pos_TP+pos_FP)
        matrix = {'pos_FN': pos_FN, 'pos_FP': pos_FP, 'pos_TN': pos_TN, 
                'pos_TP': pos_TP, 'pos_acc': pos_acc, 'pos_precision': pos_precision,
                'conf_FN': conf_FN, 'conf_FP': conf_FP, 'conf_TN': conf_TN, 
                'conf_TP': conf_TP, 'conf_acc': conf_acc, 'conf_precision': conf_precision,
                'threshold>0.8 rate':pred_ball_count/ball_count}
        print(matrix)
        # val_confusion_matrix = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\val_confusion_matrix\matrix.csv'
        val_confusion_matrix = r'/usr/src/datasets/tracknet/val_confusion_matrix/matrix.csv'
        csv_path = pathlib.Path(val_confusion_matrix)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, mode='w', newline='') as file:
            fieldnames = matrix.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(matrix)

        print(f"avg predict time: { elapsed_times / len(dataloader):.2f} 毫秒")
    
    elif arg.mode == 'predict_with_hit':
        model, _ = attempt_load_one_weight(arg.model_path)
        worker = 0
        if torch.cuda.is_available():
            model.cuda()
            worker = 1
        dataset = TrackNetTestDataset(root_dir=arg.source)
        dataloader = build_dataloader(dataset, arg.batch, worker, shuffle=False, rank=-1)
        overrides = overrides.copy()
        overrides['save'] = False
        predictor = TrackNetPredictor(overrides=overrides)
        predictor.setup_model(model=model, verbose=False)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), bar_format=TQDM_BAR_FORMAT)
        elapsed_times = 0.0

        metrics = []
        for i, batch in pbar:
            # target = batch['target'][0]
            input_data = batch['img']
            idx = np.random.randint(0, 10)
            idx = 5
            # hasBall = target[idx][1].item()
            # t_x = target[idx][2].item()
            # t_y = target[idx][3].item()
            # xy = [(t_x, t_y)]
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            start_time = time.time()

            # [1*1*60*20*20]
            # [6*20*20]
            pred = predictor.inference(input_data)[0][0]
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            elapsed_times+=elapsed_time
            pbar.set_description(f'{elapsed_times / (i+1):.2f}  {i+1}/{len(pbar)}')
            
            pred_distri, pred_scores, pred_hits = torch.split(pred, [40, 10, 10], dim=0)
            pred_distri = pred_distri.reshape(4, 10, 20, 20)
            pred_pos, pred_mov = torch.split(pred_distri, [2, 2], dim=0)

            pred_pos = pred_pos.permute(1, 0, 2, 3).contiguous()
            pred_mov = pred_mov.permute(1, 0, 2, 3).contiguous()

            pred_pos = torch.sigmoid(pred_pos)
            pred_scores = torch.sigmoid(pred_scores)
            pred_mov = torch.tanh(pred_mov)


            for frame_idx in range(10):
                p_conf = pred_scores[frame_idx]
                p_cell_x = pred_pos[frame_idx][0]
                p_cell_y = pred_pos[frame_idx][1]

                max_position = torch.argmax(p_conf)
                # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
                max_y, max_x = np.unravel_index(max_position.cpu().numpy(), p_conf.shape)
                max_conf = p_conf[max_y, max_x]

                metric = {}
                metric["grid_x"] = max_x
                metric["grid_y"] = max_y
                metric["x"] = p_cell_x[max_y][max_x]
                metric["y"] = p_cell_y[max_y][max_x]
                metric["conf"] = max_conf

                if i == 0 or frame_idx == 9:
                    metrics.append(metric)
                elif i > 0 and metric["conf"] > metrics[i+frame_idx]["conf"]:
                    metrics[i+frame_idx] = metric

            first_metric = metrics[i]
            display_predict_image(
                    input_data[0][0],  
                    [first_metric], 
                    str(i),
                    )

        print(f"avg predict time: { elapsed_times / len(dataloader):.2f} 毫秒")    
    elif arg.mode == 'train_v2':
        model = TrackNet(overrides)
        model.train()
    elif arg.mode == 'train_v3':
        model = TrackNet(overrides)
        model.train(freeze_layers=8)
    elif arg.mode == 'val_v2':
        model = TrackNet(overrides)
        model.val()
    elif arg.mode == 'predict_v2':
        """
        影片預測模式
        支援：
        1. 單一影片：--source "/path/to/video.mp4"
        2. 影片資料夾：--source "/path/to/videos/"
        """
        source_path = Path(arg.source)
    
        if source_path.is_file():
            video_files = [source_path]
            LOGGER.info(f"Processing single video: {source_path}")
        
        elif source_path.is_dir():
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.MP4', '.AVI']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(source_path.glob(f'*{ext}'))
            
            if not video_files:
                LOGGER.error(f"No video files found in: {source_path}")
                return
            
            video_files = sorted(video_files)
            LOGGER.info(f"Found {len(video_files)} videos in: {source_path}")
        
        else:
            LOGGER.error(f"Invalid source: {source_path} (not a file or directory)")
            return
        
        for video_idx, video_path in enumerate(video_files, 1):
            LOGGER.info("=" * 70)
            LOGGER.info(f"Processing video {video_idx}/{len(video_files)}: {video_path.name}")
            LOGGER.info("=" * 70)
            
            try:
                LOGGER.info(f"Using video mode: {video_path}")
                
                paths = parse_video_path(str(video_path), 'runs/detect/predict_temp')
                
                os.makedirs(os.path.dirname(paths['output_video']), exist_ok=True)
                os.makedirs(os.path.dirname(paths['output_csv']), exist_ok=True)
                os.makedirs(paths['output_frame_runs'], exist_ok=True)
                
                dataset = TrackNetVideoDataset(
                    video_path=str(video_path),
                    num_input=10,
                    imgsz=640,
                    stride=10,
                    background_method=args.background_method,
                    save_raw_frames=arg.save_raw_frames,
                    raw_frame_dir=str(paths['output_frame_datasets']) if arg.save_raw_frames else None
                )
                
                video_info = dataset.get_video_info()
                LOGGER.info(f"Video: {video_info['total_frames']} frames, "
                        f"{video_info['fps']:.2f} FPS, "
                        f"{video_info['width']}x{video_info['height']}")
                
                if video_idx == 1:
                    model, _ = attempt_load_one_weight(arg.model_path)
                    if torch.cuda.is_available():
                        model.cuda()
                        LOGGER.info("Using CUDA")
                    else:
                        LOGGER.info("Using CPU")
                
                from torch.utils.data import DataLoader
                
                def video_collate_fn(batch):
                    item = batch[0]
                    if item is None:  
                        return None
                    path, img_tensor, frames_color, vid_cap, fids, timestamps = item
                    return (
                        [path],
                        img_tensor.unsqueeze(0),
                        frames_color,
                        [vid_cap],
                        fids,
                        timestamps
                    )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=video_collate_fn
                )
                
                overrides_copy = overrides.copy()
                overrides_copy['save'] = True
                overrides_copy['project'] = 'runs/detect'
                overrides_copy['name'] = 'predict'
                overrides_copy['exist_ok'] = False
                
                predictor = TrackNetPredictor(
                    overrides=overrides_copy,
                    use_nms=arg.use_nms,
                    video_info=video_info,
                    conf_threshold=arg.conf,
                    trail_length=arg.trail_length,
                )
                predictor.setup_model(model=model, verbose=(video_idx == 1))  
                
                if video_idx == 1:
                    predictor.save_dir = predictor.get_save_dir()
                    base_save_dir = predictor.save_dir
                    LOGGER.info(f"Save directory: {predictor.save_dir}")
                else:
                    predictor.save_dir = base_save_dir
                
                paths = parse_video_path(str(video_path), str(predictor.save_dir))
                os.makedirs(os.path.dirname(paths['output_video']), exist_ok=True)
                os.makedirs(os.path.dirname(paths['output_csv']), exist_ok=True)
                os.makedirs(paths['output_frame_runs'], exist_ok=True)

                trail_frame_dir = os.path.join(str(predictor.save_dir), paths['parent_dir'], 'frame_trail', paths['video_name']) if arg.save_trail_frames else None
                if trail_frame_dir:
                    os.makedirs(trail_frame_dir, exist_ok=True)
                predictor.frame_save_dir = trail_frame_dir
                
                predictor.setup_video_writer(
                    output_path=paths['output_video'],
                    fps=video_info['fps'],
                    width=video_info['width'],
                    height=video_info['height']
                )
                
                LOGGER.info(f"Starting prediction on {len(dataloader)} batches...")
                pbar = tqdm(dataloader, desc=f"[{video_idx}/{len(video_files)}] {video_path.name}", total=len(dataloader))
                
                for batch_idx, batch in enumerate(pbar):
                    if batch is None:
                        continue
                    path, im0s, frames_color, vid_cap, fids, timestamps = batch
                    
                    if batch_idx == 0:
                        LOGGER.info(f"Data check - fids: {len(fids)}, frames: {len(frames_color)}, timestamps: {len(timestamps)}")
                    
                    if torch.cuda.is_available():
                        im0s = im0s.cuda()
                    
                    im = predictor.preprocess(im0s)
                    preds = predictor.inference(im)
                    results = predictor.postprocess_video(
                        preds, im, im0s, 
                        fids,
                        timestamps,
                        frames_color
                    )
                    
                    pbar.set_postfix({
                        'batch': f'{batch_idx+1}/{len(dataloader)}',
                        'predictions': len(predictor.csv_rows)
                    })
                
                predictor.cleanup_video_writer()
                predictor.save_csv_results(paths['output_csv'])
                
                LOGGER.info(f"✓ Video {video_idx}/{len(video_files)} complete:")
                LOGGER.info(f"  Output video: {paths['output_video']}")
                LOGGER.info(f"  Output CSV: {paths['output_csv']}")
                LOGGER.info(f"  Total predictions: {len(predictor.csv_rows)}")
                
            except Exception as e:
                LOGGER.error(f"✗ Failed to process {video_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue  
        
        LOGGER.info("=" * 70)
        LOGGER.info(f"Batch processing complete! Processed {len(video_files)} videos")
        LOGGER.info("=" * 70)


def confusion_matrix_gpu(y_true, y_pred):
    conf_matrix = torch.zeros(2, 2, dtype=torch.int64, device=y_true.device)
    
    conf_matrix[0, 0] = torch.sum((y_true == 0) & (y_pred == 0))  # True Negative (TN)
    conf_matrix[0, 1] = torch.sum((y_true == 0) & (y_pred == 1))  # False Positive (FP)
    conf_matrix[1, 0] = torch.sum((y_true == 1) & (y_pred == 0))  # False Negative (FN)
    conf_matrix[1, 1] = torch.sum((y_true == 1) & (y_pred == 1))  # True Positive (TP)
    
    return conf_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a custom model with overrides.')

    parser.add_argument('--model_path', type=str, default=r'/usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml', help='Path to the model')
    parser.add_argument('--mode', type=str, default='train_v2', help='Mode for the training (e.g., train, test)')
    parser.add_argument('--data', type=str, default='tracknet.yaml', help='Data configuration (e.g., tracknet.yaml)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--plots', type=bool, default=False, help='Whether to plot or not')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--source', type=str, default=r'/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/', help='testvideo-source')
    parser.add_argument('--val', type=bool, default=True, help='run val')
    parser.add_argument('--use_dxdy_loss', action=argparse.BooleanOptionalAction, default=True, help='use dxdy loss or not')
    parser.add_argument('--use_resampler', action=argparse.BooleanOptionalAction, default=True, help='use resampler on each epoch')


    parser.add_argument('--use_nms', action='store_true',  help='Use NMS for multi-ball detection')
    parser.add_argument('--save_raw_frames', action='store_true', help='Save raw frames to datasets/.../frame/')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--background_method', type=str, default='mean', choices=['none', 'median', 'mean'], help='background remove method')
    
    parser.add_argument('--use_downsample', action=argparse.BooleanOptionalAction, default=True, help='Enable frame rate downsampling augmentation')
    parser.add_argument('--ds_min_fps', type=int, default=30, help='Minimum fps floor after downsampling')
    parser.add_argument('--ds_maxstep', type=int, default=2, help='Maximum downsample step allowed')

    # predict_v2 相關參數
    parser.add_argument('--trail_length', type=int, default=0, help='顯示歷史軌跡點數量')
    parser.add_argument('--save_trail_frames', action='store_true', help='將含軌跡的每幀存成圖片')


    args = parser.parse_args()
    # args.epochs = 50
    # for val
    # args.batch = 1
    # args.mode = 'val_v2'
    # args.mode = 'predict_v2'
    # args.model_path = r'/Users/bartek/git/BartekTao/ultralytics/runs/detect/train178/weights/last.pt'
    main(args)