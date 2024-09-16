# ===========================================================================================
# Copyright (C) 2024 @Haishan Liu, hliu240@ucr.edu
#
# This file contains the key functions to get fused detection from the camera and the LiDAR data
# ==============================================================================================


import cv2
import os
import numpy as np
import pandas as pd
from mmdet.apis import DetInferencer
from datetime import datetime, timedelta
from ISC_utils import *
from ISC_sensor_fusion import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq


# cameras activate later than the lidar
# for all cameras, get the max timestamp to sync
CLUSTER_LABELS = {i: object for i, object in enumerate(mmdet.evaluation.functional.coco_classes())}
# 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 7: 'truck'
TAGRETS_LABELS = set([0, 1, 2, 3, 5, 7])

CAM1_MASK = np.load('./ISC_mask/Camera1_mask.npy')
CAM2_MASK = np.load('./ISC_mask/Camera2_mask.npy')
CAM3_MASK = np.load('./ISC_mask/Camera3_mask.npy')
CAM4_MASK = np.load('./ISC_mask/Camera4_mask.npy')
CAM5_MASK = np.load('./ISC_mask/Camera5_mask.npy')

name_to_bbox_size = {
    'VRU_Adult_Using_Motorized_Bicycle': [0.96, 1.64, 1.65],
    'Passenger_Vehicle': [2.60, 5.08, 1.85],
    'VRU_Child': [0.68, 0.65, 1.08],
    'VRU_Adult': [0.83, 0.75, 1.59],
    'VRU_Adult_Using_Cane': [0.75, 0.69, 1.45],
    'VRU_Adult_Using_Manual_Scooter': [1.13, 1.23, 1.76],
    'VRU_Adult_Using_Crutches': [1.14, 1.09, 2.08],
    'VRU_Adult_Using_Cardboard_Box': [1.11, 1.15, 1.68],
    'VRU_Adult_Using_Walker': [0.99, 1.12, 1.58],
    'VRU_Adult_Using_Manual_Wheelchair': [1.08, 1.20, 1.28],
    'VRU_Adult_Using_Stroller': [0.97, 1.56, 1.68],
    'VRU_Adult_Using_Skateboard': [1.31, 1.71, 1.40],
    'VRU_Adult_Using_Manual_Bicycle': [0.89, 1.53, 1.38],
    }
    

    
def test_clsuter_center(Run_num, sync_one_frame = False, selected_frame = None, sync_one_cam = False, selected_cam = None, sync_all = False):
    ''' Test the synchronization between the camera and LiDAR data. The function
        will display the camera image and the LiDAR points that are projected onto it
        Inputs: Run_num - the run number to test
                sync_one_frame - if True, the function will test the cluster of one frame, please specify the selected_frame
                sync_one_cam - if True, the function will test the cluster of one camera across all frames, please specify the selected_cam,
                sync_all - if True, the function will test the cluster of all cameras across all frames'''
    function_dict  = {1: get_lidar2_camera1_projection_matrix,
                    2: get_lidar2_camera2_projection_matrix,
                    3: get_lidar2_camera3_projection_matrix,
                    4: get_lidar2_camera4_projection_matrix,
                    5: get_lidar2_camera5_projection_matrix,
                    6: get_lidar2_camera6_projection_matrix,
                    7: get_lidar2_camera7_projection_matrix,
                    8: get_lidar2_camera8_projection_matrix} 
    

    # before running, we need to config the locations
    method_config = 'dab-detr_r50_8xb2-50e_coco'
    inferencer = DetInferencer(method_config)
    dataset_dir = '../datasets/validation_data_full'
    run_folder_dir = f'{dataset_dir}/Run_{Run_num}'
    
    # saving dir
    fusion_dir = f'{run_folder_dir}/fusion_cluster_v2' # save the fusion result to visulize some frames to a folder
    fusion_label_dir = f'{run_folder_dir}/masked_fusion_label_coco_v2' # save the fusion result to a folder, _coco not convert the label. just use coco label
    os.makedirs(fusion_dir, exist_ok=True)
    os.makedirs(fusion_label_dir, exist_ok=True)

    video_format = f'VisualCamera{{}}_Run_{Run_num}.mp4'
    syn_csv_format = f'VisualCamera{{}}_Run_{Run_num}_frame-timing_sync.csv'
    lidar_path_format = f'{dataset_dir}/filtered_bin_lidar12/Run_{Run_num}/Lidar12_bin_filtered/strongest/{{:06d}}.bin'
    
    # loop through all the cameras to get the first and last synchronized frame for the lidar
    # camera startes later than the lidar, may end earlier or later than the lidar
    first_syn_lidar_frames = []
    last_syn_lidar_frames = []
    syn_dict = {}
    ISC_select_cams = [2,4]
    for i in ISC_select_cams:
        csv_path = os.path.join(run_folder_dir, syn_csv_format.format(i))
        if not os.path.exists(csv_path):
            print(f'Error: {csv_path} does not exist')
            continue
        # drop the nan value, in the  Lidar2_frame_idx column
        syn_time_framing = pd.read_csv(csv_path)
        # syn_time_framing['Lidar2_frame_idx'] = syn_time_framing['Lidar2_frame_idx'].astype(int)
        syn_dict[i] = syn_time_framing # save the dataframe to a dictionary for later use
        first_syn_lidar_frames.append(syn_time_framing['Lidar2_frame_idx'].min())
        last_syn_lidar_frames.append(syn_time_framing['Lidar2_frame_idx'].max())

    first_syn = max(first_syn_lidar_frames)
    last_syn = min(last_syn_lidar_frames)

    # The scenario is: with all synced cameras and the lidar frame, we want to get the detection from the camera and the lidar points#
    
    if sync_one_frame:
        syn_bins = [selected_frame]
        cameras = ISC_select_cams # define the cameras to be used
        # cameras = np.arange(1,6) # define the cameras to be used


    if sync_one_cam:
        # i should be the camera number in the for loop
        # get the width and height of the original camera
        video_path = os.path.join(run_folder_dir, video_format.format(selected_cam))
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Unable to open video {video_path}")
            raise ValueError(f"Unable to open video {video_path}")
            # capture the syn frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create a video writer to save the images the same size as the original camera
        out = cv2.VideoWriter(f'{fusion_dir}/Camera{selected_cam}_all.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        syn_bins = np.arange(first_syn, last_syn+1).astype(int)
        cameras = [selected_cam]
    
    # TODO: add the sync_all option, write the video to the fusion folder
    if sync_all:
        syn_bins = np.arange(first_syn, last_syn+1).astype(int)
        cameras = ISC_select_cams # define the cameras to be used
        print(f'cameras {cameras}')
        out_writers = {}
        for i in cameras:
            video_path = os.path.join(run_folder_dir, video_format.format(i))
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Unable to open video {video_path}")
                raise ValueError(f"Unable to open video {video_path}")
                # capture the syn frame
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # create a video writer to save the images the same size as the original camera
            out = cv2.VideoWriter(f'{fusion_dir}/Camera{i}_all.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
            out_writers[i] = out

        # out = cv2.VideoWriter(f'{fusion_dir}/Camera{selected_cam}_all.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (, height))
    
    run_fused_result = {} # fused from raw lidar and cameras
    refined_fused_result = {} # fused from conventional pipeline and the fused result
    history_object = []
    # 解决camera 漏检的问题， 如果camera 没有检测到，赋予之前的值
    heapq.heapify(history_object) # store the history of the fused result (bin_idx, centers)
    
    conventional_df = load_conventional_pipeline_result(Run_num)
    
    # not sycned  with the cameras
    for bin_idx in range(0, int(first_syn)):
        conventional_result = get_one_bin_detection(conventional_df, f'{bin_idx:06d}.pcd')
        convention_centers = conventional_result[['x', 'y', 'z', 'label','score','rotation']].values
        refined_fused_result[bin_idx] = convention_centers

    for bin_idx in tqdm(syn_bins):
        print(f'-------Processing lidar frame {bin_idx}------')
        # get the frame from the lidar2
        lidar2_frame_path = lidar_path_format.format(bin_idx)
        # work on camera 4 first
        # create eight subplots to show the images each ax with size 10,5 
        if sync_one_frame:
            fig, axs = plt.subplots(3,2 , figsize=(20, 10))
        velo_image_list = []

        # record the proposal from each camera
        camera_proposal = {}
        for i in cameras:
            syn_time_framing = syn_dict[i]
        
            idx = syn_time_framing[syn_time_framing['Lidar2_frame_idx'] == bin_idx].index.max()
            if np.isnan(idx):
                # print(f'Camera {i} frame {bin_idx} is not available')
                # try to find the nearest frame
                idx_min = syn_time_framing[syn_time_framing['Lidar2_frame_idx'] == bin_idx - 1].index.max()
                idx_max = syn_time_framing[syn_time_framing['Lidar2_frame_idx'] == bin_idx + 1].index.min()
                if np.isnan(idx_min) and not np.isnan(idx_max):
                    idx = idx_max
                elif not np.isnan(idx_min) and np.isnan(idx_max):
                    idx = idx_min
                elif not np.isnan(idx_min) and not np.isnan(idx_max):
                    idx = (idx_min + idx_max) // 2
                else:
                    print(f'Error: Camera {i} frame for Lidar {bin_idx} is not available')
                    continue  
         
            image_number = syn_time_framing.loc[idx, 'Image_number']
            print(f'Camera {i} frame {image_number} is syn with lidar frame {bin_idx}')
        
            video_path = os.path.join(run_folder_dir, video_format.format(i))
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Unable to open video {video_path}")
                continue
            # capture the syn frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, image_number)
            ret, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not ret:
                print(f"Unable to read the {image_number} frame from video {video_path}")
                cap.release()
                continue
          
            T_velo_cam, translation_velo_cam = function_dict[i]()
            K = get_single_camera_intrinsic(camera_number= i)
            D = get_single_camera_distortion(camera_number= i)
            
        
            # get the inverse of the transformation matrix
            T_velo_cam_homogeneous = np.vstack((T_velo_cam, (0, 0, 0, 1)))
            T_cam_velo = np.linalg.inv(T_velo_cam_homogeneous)[:3, :]
            
            # Add mask to the image to aviod the detection outside intersection area
            # print(frame.shape)
            frame = (frame * eval(f'CAM{i}_MASK')[:,:,None]).astype(np.uint8)
            # print(frame.shape)
            # plt.imshow(frame)
            # plt.show()

            if not os.path.exists(lidar2_frame_path):
                print(f'Error: {lidar2_frame_path} does not exist')
                raise ValueError(f'Error: {lidar2_frame_path} does not exist')
                continue

            bboxes, labels, scores, clusters, centers, velo_uvz = get_fused_detection(image= frame, 
                                                                        bin_path= lidar2_frame_path, 
                                                                        model= inferencer, K= K, D= D, 
                                                                        T_velo_cam= T_velo_cam, T_cam_velo = T_cam_velo, draw_depth= False)
           
            # print('centers shape', centers.shape)
            clusters = center_empty_filter(clusters)
            if len(clusters) == 0:
                print(f'No cluster is detected in the lidar frame {bin_idx}')
                print('\n')
                if sync_one_frame:
                    ax = axs[(i-1)//2, (i-1)%2]
                    ax.imshow(frame)
                    ax.set_title(f'Camera {i} frame {image_number}')
                    plt.imsave(os.path.join(fusion_dir, f'Run_{Run_num}_Lidar_frame{bin_idx}_Camera{i}_frame_{image_number}.jpg'), frame)
                if sync_all:
                    cv2.putText(frame, f'Lidar frame {bin_idx} Camera {i} frame {image_number}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    out_writers[i].write(frame)
                continue
            
            # input to the draw_velo_on_image is a 3xN array

            # filter out the empty centers
            centers = [center for center in centers if len(center) > 0]
            camera_proposal[i] = centers
            velo_image = draw_cluster_on_image(clusters.T, frame)
            velo_image_list.append(velo_image)
            print('center', centers)
            print('center shape', len(centers)) 
            print('\n')
            
            ### ========== code below is for visualization only ==========
            if sync_one_cam:
                # put text on the image
                cv2.putText(velo_image, f'Lidar frame {bin_idx} Camera {i} frame {image_number}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(velo_image)

            elif sync_one_frame:
                ax = axs[(i-1)//2, (i-1)%2]
                ax.imshow(velo_image)
                ax.set_title(f'Camera {i} frame {image_number}')
                plt.imsave(os.path.join(fusion_dir, f'Run_{Run_num}_Lidar_frame{bin_idx}_Camera{i}_frame_{image_number}.jpg'), velo_image)
            
            elif sync_all:
                cv2.putText(velo_image, f'Lidar frame {bin_idx} Camera {i} frame {image_number}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out_writers[i].write(velo_image)
            # plt.imshow(velo_image)
        if sync_one_frame:
            plt.suptitle(f'Run {Run_num} Lidar frame {bin_idx}')
            plt.tight_layout()
            # plt.imsave(f'{fusion_dir}/Lidar_frame{bin_idx}_summary_new.jpg', big_image)
            plt.savefig(f'{fusion_dir}/Lidar_frame{bin_idx}_summary.jpg')
            try:
                big_image = concatenate_images(velo_image_list, (3, 2))
                cv2.imwrite(f'{fusion_dir}/Lidar_frame{bin_idx}_summary_new.jpg', big_image)
            # plt.imshow(velo_image)
            except ValueError:
                print(f'Error: can not combine the images for lidar frame {bin_idx}')
        ### ========== code above is for visualization only ==========
        # run_fused_result stores the history of the fused result
        voted_camera_proposal = camera_proposal_decision(camera_proposal)
        print('voted camera proposal: \n', voted_camera_proposal)
        run_fused_result[bin_idx] = voted_camera_proposal

        # convention_fused_proposal = refine_conventional_pipeline_result(conventional_df, bin_idx, voted_camera_proposal)
        # refined_fused_result[bin_idx] = convention_fused_proposal
        # print('convention_fused_proposal: \n', convention_fused_proposal)


        ## ========= code below is for result fusion ==========
    if sync_all:    
        write_out_fused_result(run_fused_result, fusion_label_dir, Run_num)
        for i in cameras:
            out_writers[i].release()
        # print(f'Camera {i} frame {bin_idx} is processed successfully')

if __name__ == '__main__':
    ### use mode 1: check for one frame in one Run ###
    # test_clsuter_center(Run_num=48, sync_one_frame = True, selected_frame = 229)
   

    ### use mode 2: check for one camera in one Run ###
    # test_clsuter_center(Run_num=55, sync_one_cam=True, selected_cam = 5)
    

    ### use mode 3: check for all selected cameras in one Run ###
    # test_clsuter_center(Run_num=1003, sync_all= True)

    # use mode 4:function to get the fused result for all the runs ###
    run_nums = [int(dir.split('_')[-1])for dir in os.listdir('../datasets/validation_data_full') if dir.startswith('Run_')]
    error_run_nums = [] 
    for run_num in tqdm(run_nums):
        if run_num == 55:
            continue
        try:
            test_clsuter_center(Run_num=run_num, sync_all= True)
        except Exception as e:
            error_run_nums.append(run_num)
            print(f'Error: {e}')
            continue
    print('Error run numbers', error_run_nums)