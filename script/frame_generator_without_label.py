import cv2
import csv
import os
import sys
import shutil
from glob import glob
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description = 'Frame Generator Without Label')
parser.add_argument('--dataset_folder', type=str, required=True, help = 'Dataset Folder (Include Different Locations)')
parser.add_argument('--locations', type=lambda s: s.split(' '), required=True, help = 'Example: --location EC234 nctu_old_gym profession_dataset (Use 3 locations)')
args = parser.parse_args()


HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag

# get all data folder under locations
game_list = []
for location in args.locations:
    if os.path.isdir(os.path.join(args.dataset_folder,location)):
        game_list.append(os.path.join(args.dataset_folder,location))

for game in game_list:
    p = os.path.join(game, 'video', '*')
    video_list = glob(p)
    frame_folder = os.path.join(game, 'frame')
    # heatmap_folder = os.path.join(game, 'heatmap')
    # if not os.path.isdir(heatmap_folder):
    #     os.makedirs(heatmap_folder)
    if not os.path.isdir(frame_folder):
        os.makedirs(frame_folder)
    for videoName in video_list:
        print(videoName)
        rallyName = os.path.splitext(os.path.basename(videoName))[0]
        # rallyCSV = os.path.join(game, 'csv', rallyName + '_ball.csv')
        # df = pd.read_csv(rallyCSV)
        print('Processing: ', rallyName)

        video_frame_folder = os.path.join(frame_folder, rallyName)
        # video_heatmap_folder = os.path.join(heatmap_folder, rallyName)

        if not os.path.isdir(video_frame_folder):
            if os.path.isdir(video_frame_folder):
                shutil.rmtree(video_frame_folder)
            # if os.path.isdir(video_heatmap_folder):
            #     shutil.rmtree(video_heatmap_folder)
            cap = cv2.VideoCapture(videoName)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # truncating data, only from 1/8 rally time before first visible frame to 1/8 rally time after last visible frame
            # visible_frames_cnt = len(df[df['Visibility'] == 1])
            # if visible_frames_cnt == 0:
            #     print('No visible frame in this rally')
            #     # remove video and csv
            #     os.remove(videoName)
            #     os.remove(rallyCSV)
            #     continue
            # first_visible_index = df[df['Visibility'] == 1].index[0]
            # last_visible_index = df[df['Visibility'] == 1].index[-1]

            # offset = visible_frames_cnt // 10

            # start_index = max(0, first_visible_index)
            # end_index = min(frame_count - 1, last_visible_index)

            start_index = 0
            end_index = frame_count - 1
            # print('Start Index: ', start_index)
            # print('End Index: ', end_index)
            
            if not os.path.isdir(video_frame_folder):
                os.makedirs(video_frame_folder)
            # if not os.path.isdir(video_heatmap_folder):
            #     os.makedirs(video_heatmap_folder)

            # capture video frames and save as image from specified range of video
            cap = cv2.VideoCapture(videoName)
            success, count, idx = True, 0, 0
            success, image = cap.read()

            while success:
                if count >= start_index and count <= end_index:
                    cv2.imwrite(os.path.join(video_frame_folder, '{}.png'.format(idx)) , image)
                    # if count < len(df) and df['Visibility'].values[count] == 1:
                    #     heatmap = genHeatMap(WIDTH, HEIGHT, int(df['X'].values[count] / image.shape[1] * WIDTH), int(df['Y'].values[count] / image.shape[0] * HEIGHT), sigma, mag)
                    # else:
                    #     heatmap = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
                    # cv2.imwrite(os.path.join(video_heatmap_folder, '{}.png'.format(idx)), heatmap * 255)
                    idx += 1
                count += 1
                success, image = cap.read()

print('finish')