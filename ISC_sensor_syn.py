import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ISC_utils import *


'''
usage: python ISC_sensor_syn.py

1. configure the datset_dir: contains the validation/test data, Run_1, Run_2, ...
2. confirm the folder_dir -- Run_1, Run_2, ...
3. Confirm the csv_format and syn_csv_format, and Lidar2 frame timing

'''
def SynCamToLidar(Run_num):
    # create a txt to save the sync log
    
    # Folder and video configuration
    dataset_dir = '../datasets/validation_data_full'
    folder_dir = f'{dataset_dir}/Run_{Run_num}'
    f = open(f'{dataset_dir}/Run_{Run_num}/sync_log.txt', 'w')
    f.write(f'Sync all cameras to the Lidar2 frame timing\n')

    cam_csv_format = f'VisualCamera{{}}_Run_{Run_num}_frame-timing.csv'
    syn_csv_format = f'VisualCamera{{}}_Run_{Run_num}_frame-timing_sync.csv'
    lidar2_csv_format = f'Lidar2_Run_{Run_num}_frame-timing.csv'

    # ------------------ Get the Lidar 2 frame timing from the all_timing.csv ------------------
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
        lidar2_frame_timing.append({'Lidar2': lidar2_start_time + timedelta(milliseconds=i*100), 'bin_idx': i})

    lidar2_frame_timing_df = pd.DataFrame(lidar2_frame_timing)
    lidar2_frame_timing_df.to_csv(os.path.join(folder_dir, lidar2_csv_format), index=False)
    print(f'Lidar2 frame timing is saved successfully')

    # start processing the cameras, save the sync_frame_timing to the csv file
    for i in range(1, 9):
        csv_path = os.path.join(folder_dir, cam_csv_format.format(i))
        # check is the file exists
        if not os.path.exists(csv_path):
            f.write(f'Camera {i} frame timing is not found\n')
            print(f'Camera {i} frame timing is not found')
            continue    
        camera_frame_timing = pd.read_csv(csv_path)
        def find_nearest_lidar2_frame(x):
            if (x - lidar2_frame_timing_df['Lidar2']).abs().min() > timedelta(seconds=0.03):
                return np.nan
            return (x - lidar2_frame_timing_df['Lidar2']).abs().idxmin()
        # for each camera frame, find the nearest lidar2 frame  
        nearest_idx = pd.to_datetime(camera_frame_timing['Timestamp'], format="%Y-%m-%d-%H-%M-%S_%f").apply(find_nearest_lidar2_frame)
        # print(nearest_idx)
        # save the nearest index to the camera frame timing
        camera_frame_timing['Lidar2_frame_idx'] = nearest_idx
        camera_frame_timing.to_csv(os.path.join(folder_dir, syn_csv_format.format(i)), index=False)
        f.write(f'Camera {i} frame timing is saved successfully\n')
        print(f'Camera {i} frame timing is saved successfully')

if __name__ == '__main__':
    # get the run_nums
    dataset_dir = '../datasets/validation_data_full'
    run_nums = [int(dir.split('_')[-1]) for dir in os.listdir(dataset_dir) if 'Run_' in dir]
    print(run_nums)
    print('total runs:', len(run_nums))
    error_runs = []
    for Run_num in run_nums:
        try:
            SynCamToLidar(int(Run_num))
            print(f'Run {Run_num} is processed successfully')
        except Exception as e:
            # print(f'Run {Run_num} is failed to process')
            error_runs.append(Run_num)
            print(e)
            continue
    print('error runs:', error_runs)