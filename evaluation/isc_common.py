import numpy as np
import os
import pandas as pd

def get_label_anno(one_frame_df):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    length = one_frame_df.shape[0]
 
    # print(content)
    annotations['name'] = one_frame_df['label'].values
    annotations['truncated'] = np.zeros((length), dtype=float)
 
    annotations['occluded'] = np.zeros((length), dtype=int)
    annotations['alpha'] = np.zeros((length), dtype=float)
    annotations['bbox'] = np.zeros((length, 4), dtype=float)
    annotations['dimensions'] = one_frame_df[['x_len', 'y_len', 'z_len']].values.reshape(-1, 3)
    annotations['location'] = one_frame_df[['x_ctr', 'y_ctr', 'z_ctr']].values.reshape(-1, 3)

    annotations['rotation_y'] = one_frame_df['z_rot'].values
    
    if 'score' in one_frame_df.columns:
        annotations['score'] = one_frame_df['score'].values
    else:
        annotations['score'] = np.zeros((length), dtype=float)
    return annotations

def get_label_annos(Run_num):
    # dataset_dir = '/home/haishan/Projects/Intersection_safety/Intersection_Safety_Challenge/datasets/validation_data_full'
    gt_label_format = f'./kitti_GT_2/Run_{Run_num}/Run_{Run_num}_GT.txt'

    dt_dir = '../camera_fused_label/fused_label_lidar12_cam24/masked_fusion_label_coco_v2'
    dt_label_format = f'{dt_dir}/Run_{Run_num}_detections_fusion_lidar12_camera_search-based_eval.csv'

    # need to assert gt_df.shape[0] == dt_df.shape[0]
    
    gt_df = pd.read_csv(gt_label_format, header=None, sep=',')
  
    dt_df = pd.read_csv(dt_label_format, header=None, sep=',')
    # add the header to the dataframe
   
    gt_df.columns = ['bin_indx', 'label', 'x_ctr', 'x_len', 'y_ctr', 'y_len', 'z_ctr', 'z_len', 'z_rot'] # gt has no score
    
    dt_df.columns = ['bin_indx', 'label', 'x_ctr', 'x_len', 'y_ctr', 'y_len', 'z_ctr', 'z_len', 'z_rot', 'score']

    # print(df.head(5))
    
    # for each frame, 
    gt_unique_frames = gt_df['frame'].unique()
    dt_unique_frames = dt_df['frame'].unique()
    # Ensure both dataframes have matching frames by filtering based on common frames
    common_frames = np.intersect1d(gt_unique_frames, dt_unique_frames)
    
    gt_df = gt_df[gt_df['frame'].isin(common_frames)]
    dt_df = dt_df[dt_df['frame'].isin(common_frames)]
    
    gt_unique_frames = gt_df['frame'].unique()  # update unique frames
    dt_unique_frames = dt_df['frame'].unique()

    # Final check to ensure frame counts match
    if len(gt_unique_frames) != len(dt_unique_frames):
        raise ValueError(f'Frame mismatch: gt has {len(gt_unique_frames)} frames, dt has {len(dt_unique_frames)} frames')
       
    gt_annos = []
    for frame in gt_unique_frames:
        one_frame_df = gt_df[gt_df['frame'] == frame]
        gt_annos.append(get_label_anno(one_frame_df))
    
    dt_annos = []
    for frame in dt_unique_frames:
        one_frame_df = dt_df[dt_df['frame'] == frame]
        dt_annos.append(get_label_anno(one_frame_df))
    return gt_annos, dt_annos


if __name__ == '__main__':
    dataset_dir = '/home/haishan/Projects/Intersection_safety/Intersection_Safety_Challenge/datasets/validation_data_full'
    run_nums = [int(dir.split('_')[-1]) for dir in os.listdir(dataset_dir) if dir.startswith('Run_')]
    print(run_nums)
    full_gt_annos = []
    full_dt_annos = []
    for run_num in run_nums:
        gt_annos, dt_annos = get_label_annos(run_num)
        full_gt_annos.extend(gt_annos)
        full_dt_annos.extend(dt_annos)
    # print(len(full_annos))
       
    # annos = get_label_annos(48, gt_anno=True) 
    # print(annos)