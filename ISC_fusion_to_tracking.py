import cv2 as cv
import os
import numpy as np

import argparse 
from ISC_utils import * # we define the fuctions to get projection matrix in the ISC_utils.py and the bbox drawing function

class KittiLabel:
    def __init__(self, label, dimensions, location, rotation_y, bbox = (0, 0, 0, 0), truncated=0, occluded=0, alpha=0 ):
        self.label = label
        self.bbox = bbox
        self.truncated = truncated
        self.occluded = occluded
        self.alpha = alpha
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.score = 0
    
    def create_one_line(self):
        return f'{self.label} {self.truncated} {self.occluded} {self.alpha} {self.bbox[0]} {self.bbox[1]} {self.bbox[2]} {self.bbox[3]} {self.dimensions[0]} {self.dimensions[1]} {self.dimensions[2]} {self.location[0]} {self.location[1]} {self.location[2]} {self.rotation_y}'

def detr_for_tracking(run_fused_file, run_sync_file, tracking_dir):
    fused_result = pd.read_csv(run_fused_file)
    cam4_sync = pd.read_csv(run_sync_file)
    unique_frames = fused_result['bin_idx'].unique()
    for frame in unique_frames:
        frame_fused_result = fused_result[fused_result['bin_idx'] == frame] 
        try:
            frame_cam4_sync = cam4_sync[cam4_sync['Lidar2_frame_idx'] == int(frame)]
            frame_cam4_sync = frame_cam4_sync['Image_number'].iloc[0]
        except IndexError:
            try:
                frame_cam4_sync = cam4_sync[cam4_sync['Lidar2_frame_idx'] == int(frame) - 1]
                frame_cam4_sync = frame_cam4_sync['Image_number'].iloc[-1]
                print(f'Frame {frame} not found in camera 4 sync file, sync to frame {int(frame) - 1}')
            except IndexError:
                print(f'Frame {frame} not found in camera 4 sync file')
                continue
    

        # save the tracking label frame by frame
        with open(f'{tracking_dir}/{int(frame_cam4_sync)}.txt', 'w') as f:
            for i, row in frame_fused_result.iterrows():
                label = row['subclass']
                dimensions = (row['x_length'], row['y_length'], row['z_length'])
                location = (row['x_center'], row['y_center'], row['z_center']) 
                rotation_y = row['z_rotation']
                oneline = KittiLabel(label, dimensions, location, rotation_y).create_one_line()
                f.writelines(oneline + '\n')
        
def main():
    dataset_dir = '../datasets/validation_data_full'
    fused_result_dir = './camera_fused_label/fused_label_lidar12_cam24/masked_fusion_label_coco'
    run_fused_file_format = f'{fused_result_dir}/Run_{{}}_detections_fusion_lidar12_camera_search-based_tracking.csv'
    cam4_sync_file_format = f'{dataset_dir}/Run_{{}}/VisualCamera4_Run_{{}}_frame-timing_sync.csv'
    tracking_dir_format = f'./fusion_for_tracking/Run_{{}}'
    
    Run_nums = sorted([int(dir.split('_')[-1])for dir in os.listdir('../datasets/validation_data_full') if dir.startswith('Run_')])
    # Run_nums = [48]
    for run_num in Run_nums:
        run_fused_file = run_fused_file_format.format(run_num)
        run_sync_file = cam4_sync_file_format.format(run_num, run_num)
        tracking_dir = tracking_dir_format.format(run_num)
        os.makedirs(tracking_dir, exist_ok=True)

        print(f'Processing Run {run_num}')
        detr_for_tracking(run_fused_file, run_sync_file, tracking_dir)

if __name__ == '__main__':
   main()