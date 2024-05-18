
import argparse
from copy import copy
import csv
import os
import time
from matplotlib import patheffects
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.predict import TrackNetPredictor
from ultralytics.tracknet.test_dataset import TrackNetTestDataset
from ultralytics.tracknet.train import TrackNetTrainer
from ultralytics.tracknet.utils.loss import TrackNetLoss
from ultralytics.tracknet.utils.plotting import display_image_with_coordinates, display_predict_image
from ultralytics.tracknet.utils.transform import target_grid
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

# from ultralytics import YOLO

# # Create a new YOLO model from scratch
# model = YOLO(r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)
# # results = model.train(data='tracknet.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

def main(arg):
    overrides = {}
    overrides['model'] = arg.model_path
    overrides['mode'] = arg.mode
    overrides['data'] = arg.data
    overrides['epochs'] = arg.epochs
    overrides['plots'] = arg.plots
    overrides['batch'] = arg.batch
    overrides['patience'] = 300
    overrides['plots'] = arg.plots
    overrides['val'] = arg.val
    overrides['use_dxdy_loss'] = arg.use_dxdy_loss

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

            # [1*50*20*20]
            p = predictor.inference(input_data)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            elapsed_times+=elapsed_time
            pbar.set_description(f'{elapsed_times / (i+1):.2f}  {i+1}/{len(pbar)}')
            # [5*20*20]
            p_check = p[0, 5*idx:5*(idx+1), :]
            p_conf = torch.sigmoid(p_check[4, :, :])
            p_cell_x = torch.sigmoid(p_check[0, :, :])
            p_cell_y = torch.sigmoid(p_check[1, :, :])

            max_position = torch.argmax(p_conf)
            max_y, max_x = np.unravel_index(max_position, p_conf.shape)
            # p_x = p_cell_x[max_x, max_y]*32
            # p_y = p_cell_y[max_x, max_y]*32
            # p_gridx = max_x*32 + p_cell_x[max_x, max_y]*32
            # p_gridy = max_y*32 + p_cell_y[max_x, max_y]*32
            display_predict_image(
                input_data[0][idx],  
                [(max_y, max_x, p_cell_x[max_y, max_x], p_cell_y[max_y, max_x], p_conf[max_y, max_x])], 
                str(i),
                )
            # x, y, gx, gy = target_grid(t_x, t_y, 32)

            # if hasBall and i%10 == 0:
            #     pos_weight = torch.tensor([400])
            #     l = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
            #     cls_targets = torch.zeros(p_conf.shape[0], p_conf.shape[1])
            #     cls_targets[x][y] = 1
            #     cls_pred = p_check[4, :, :]
            #     count_ge_05 = np.count_nonzero(p_conf >= 0.5)
            #     count_lt_05 = np.count_nonzero(p_conf < 0.5)
            #     correct = True if p_conf[x][y]>=0.5 else False
            #     loss = l(cls_pred, cls_targets).sum()
            #     loss_list = [loss.item()]
            #     loss_list.append(count_ge_05)
            #     loss_list.append(count_lt_05)
            #     loss_list.append(correct)
            #     loss_list.append(p_conf[x][y])
            #     display_image_with_coordinates(input_data[0][idx], [(x*32, y*32)], [(max_x*32, max_y*32)], str(i), loss_list)
            #end_time = time.time()
            #elapsed_time = (end_time - start_time) * 1000
            #print(f'{elapsed_time:.2f}ms')
            #elapsed_times+=elapsed_time

        print(f"avg predict time: { elapsed_times / len(dataloader):.2f} 毫秒")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a custom model with overrides.')
    
    parser.add_argument('--model_path', type=str, default=r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\tracknetv4.yaml', help='Path to the model')
    parser.add_argument('--mode', type=str, default='train', help='Mode for the training (e.g., train, test)')
    parser.add_argument('--data', type=str, default='tracknet.yaml', help='Data configuration (e.g., tracknet.yaml)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--plots', type=bool, default=False, help='Whether to plot or not')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--source', type=str, default=r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\train_data', help='source')
    parser.add_argument('--val', type=bool, default=False, help='run val')
    parser.add_argument('--use_dxdy_loss', type=bool, default=True, help='use dxdy loss or not')
    
    args = parser.parse_args()
    # args.epochs = 50
    # args.batch = 1
    # args.mode = 'predict'
    # args.model_path = r'C:\Users\user1\bartek\github\BartekTao\ultralytics\runs\detect\prod_train196\weights\best.pt'
    # args.source = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\test_data'
    main(args)