import cv2
import os
import numpy as np
import pandas as pd
from mmdet.apis import DetInferencer
from datetime import datetime, timedelta
from cam_lidar_conversion import *
from ISC_utils import *

from kitti_detection_utils import get_fused_detection, draw_velo_on_image


def SynCamToLidar(Run_num):
    # create a txt to save the sync log
    f = open(f'../datasets/validation_data_full/Run_{Run_num}/sync_log.txt', 'w')
   
    f.write(f'Sync all cameras to the Lidar2 frame timing\n')
    # Folder and video configuration
    dataset_dir = '../datasets/validation_data_full'
    folder_dir = f'{dataset_dir}/Run_{Run_num}'

    csv_format = f'VisualCamera{{}}_Run_{Run_num}_frame-timing.csv'
    syn_csv_format = f'VisualCamera{{}}_Run_{Run_num}_frame-timing_sync.csv'

    # sysnc all cameras to the same frame
    all_timing_df = pd.read_csv(f'{dataset_dir}/Run_{Run_num}/ISC_Run_{Run_num}_ISC_all_timing.csv')

    # convert the string to datetime
    lidar2_start_time = datetime.strptime(all_timing_df.iloc[-1, -3], '%Y-%m-%d %H:%M:%S.%f')
    lidar2_end_time = datetime.strptime(all_timing_df.iloc[-1, -2], '%Y-%m-%d %H:%M:%S.%f')

    # total seconds
    lidar2_duration = (lidar2_end_time - lidar2_start_time).total_seconds()
    total_frames = lidar2_duration * 10

    # create lidar2 frame timing with a 10HZ, starting from the lidar2_time
    lidar2_frame_timing = []
    for i in range(int(total_frames)):
        lidar2_frame_timing.append({'Lidar2': lidar2_start_time + timedelta(milliseconds=i*100)})

    lidar2_frame_timing_df = pd.DataFrame(lidar2_frame_timing)

    # start processing the cameras, save the sync_frame_timing to the csv file
    for i in range(1, 9):
        csv_path = os.path.join(folder_dir, csv_format.format(i))
        # check is the file exists
        if not os.path.exists(csv_path):
            f.write(f'Camera {i} frame timing is not found\n')
            print(f'Camera {i} frame timing is not found')
            continue    
        camera_frame_timing = pd.read_csv(csv_path)
        def find_nearest_lidar2_frame(x):
            if (x - lidar2_frame_timing_df['Lidar2']).abs().min() > timedelta(seconds=0.03):
                # print(f'The time difference is too large for the x {x}')
                return np.nan
            return (x - lidar2_frame_timing_df['Lidar2']).abs().idxmin()
        # for each camera frame, find the nearest lidar2 frame,  
        nearest_idx = pd.to_datetime(camera_frame_timing['Timestamp'], format="%Y-%m-%d-%H-%M-%S_%f").apply(find_nearest_lidar2_frame)
        # print(nearest_idx)
        # save the nearest index to the camera frame timing
        camera_frame_timing['Lidar2_frame_idx'] = nearest_idx
        camera_frame_timing.to_csv(os.path.join(folder_dir, syn_csv_format.format(i)), index=False)
        f.write(f'Camera {i} frame timing is saved successfully\n')
        # print(f'Camera {i} frame timing is saved successfully')

if __name__ == '__main__':
    # get the run_nums
    os.listdir('../datasets/validation_data_full')
    run_nums = [int(dir.split('_')[-1]) for dir in os.listdir('../datasets/validation_data_full') if 'Run' in dir]
    print(run_nums)
    print('total runs:', len(run_nums))
    for Run_num in run_nums:
        SynCamToLidar(int(Run_num))
        print(f'Run {Run_num} is processed successfully')