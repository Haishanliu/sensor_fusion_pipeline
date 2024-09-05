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


# cameras activate later than the lidar
# for all cameras, get the max timestamp to sync
CLUSTER_LABELS = {i: object for i, object in enumerate(mmdet.evaluation.functional.coco_classes())}
# 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
TAGRETS_LABELS = set([0, 1, 2, 3, 5, 7])

CAM1_MASK = np.load('./ISC_mask/Camera1_mask.npy')
CAM2_MASK = np.load('./ISC_mask/Camera2_mask.npy')
CAM3_MASK = np.load('./ISC_mask/Camera3_mask.npy')
CAM4_MASK = np.load('./ISC_mask/Camera4_mask.npy')
CAM5_MASK = np.load('./ISC_mask/Camera5_mask.npy')

    
def test_clsuter_center(Run_num, sync_one_frame = False, selected_frame = None, sync_one_cam = False, selected_cam = None, sync_all = False):
    ''' Test the synchronization between the camera and LiDAR data. The function
        will display the camera image and the LiDAR points that are projected onto it
        Inputs: Run_num - the run number to test
                sync_one_frame - if True, the function will test the synchronization of one frame, please specify the selected_frame
                sync_one_cam - if True, the function will test the synchronization of one camera across all frames, please specify the selected_cam'''
    function_dict  = {1: get_lidar2_camera1_projection_matrix,
                    2: get_lidar2_camera2_projection_matrix,
                    3: get_lidar2_camera3_projection_matrix,
                    4: get_lidar2_camera4_projection_matrix,
                    5: get_lidar2_camera5_projection_matrix,
                    6: get_lidar2_camera6_projection_matrix,
                    7: get_lidar2_camera7_projection_matrix,
                    8: get_lidar2_camera8_projection_matrix} 

    method_config = 'dab-detr_r50_8xb2-50e_coco'
    inferencer = DetInferencer(method_config)
    dataset_dir = '../datasets/validation_data_full'
    run_folder_dir = f'{dataset_dir}/Run_{Run_num}'
    fusion_dir = f'{run_folder_dir}/fusion_cluster' # save the fusion result to a folder
    fusion_label_dir = f'{run_folder_dir}/masked_convention_fusion_label' # save the fusion result to a folder
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
    for i in range(1, 9):
        csv_path = os.path.join(run_folder_dir, syn_csv_format.format(i))
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
        cameras = [2, 4] # define the cameras to be used
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
        cameras = [2, 4] # define the cameras to be used
        print(f'cameras {cameras}')

        # out = cv2.VideoWriter(f'{fusion_dir}/Camera{selected_cam}_all.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (, height))
    
    run_fused_result = {} # fused from raw lidar and cameras
    refined_fused_result = {} # fused from conventional pipeline and the fused result
    
    conventional_df = load_conventional_pipeline_result(Run_num)

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
                print(f'Camera {i} frame {bin_idx} is not available')
                idx_min = syn_time_framing[syn_time_framing['Lidar2_frame_idx'] == bin_idx - 1].index.max()
                
                idx_max = syn_time_framing[syn_time_framing['Lidar2_frame_idx'] == bin_idx + 1].index.min()
                idx = (idx_min + idx_max) // 2
            if np.isnan(idx):
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
            # cv2.imwrite(f'{folder_dir}/Camera{i}_frame{bin_idx}.jpg', frame)
            # cap.release()
            # get the projection matrix
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

            if sync_one_frame:
                ax = axs[(i-1)//2, (i-1)%2]
                ax.imshow(velo_image)
                ax.set_title(f'Camera {i} frame {image_number}')
                plt.imsave(os.path.join(fusion_dir, f'Run_{Run_num}_Lidar_frame{bin_idx}_Camera{i}_frame_{image_number}.jpg'), velo_image)
       
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
        voted_camera_proposal = camera_proposal_decision(camera_proposal)
        print('voted camera proposal: \n', voted_camera_proposal)
        run_fused_result[bin_idx] = voted_camera_proposal

        convention_fused_proposal = refine_conventional_pipeline_result(conventional_df, bin_idx, voted_camera_proposal)
        refined_fused_result[bin_idx] = convention_fused_proposal
        print('convention_fused_proposal: \n', convention_fused_proposal)


        ## ========= code below is for result fusion ==========
    if sync_all:    
        write_out_fused_result(run_fused_result, fusion_label_dir, Run_num)
        # print(f'Camera {i} frame {bin_idx} is processed successfully')

if __name__ == '__main__':
    ### use mode 1: check for one frame in one Run ###
    test_clsuter_center(Run_num=55, sync_one_frame = True, selected_frame = 31)
   

    ### use mode 2: check for one camera in one Run ###
    # test_clsuter(Run_num=410, sync_one_cam=True, selected_cam = 4)
    

    ### use mode 3: check for all selected cameras in one Run ###
    # test_clsuter_center(Run_num=55, sync_all= True)



    ### function to get the fused result for all the runs ###
    # run_nums = [int(dir.split('_')[-1])for dir in os.listdir('../datasets/validation_data_full') if dir.startswith('Run_')]
    # error_run_nums = [] 
    # for run_num in tqdm(run_nums):
    #     if run_num == 48:
    #         continue
    #     try:
    #         test_clsuter_center(Run_num=run_num, sync_all= True)
    #     except Exception as e:
    #         error_run_nums.append(run_num)
    #         print(f'Error: {e}')
    #         continue
    # print('Error run numbers', error_run_nums)