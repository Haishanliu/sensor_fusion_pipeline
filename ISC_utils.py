# ===================================================================================
# Copyright (C) 2024 @Haishan Liu, hliu240@ucr.edu
#
# This file contains the functions to get the transformation matrix.
# ===================================================================================

import pandas as pd
import numpy as np
import cv2 as cv
# --- The calibration module ---
## ---- define the path to the calibration files ---- ##    

cam_to_cam_file = './calibration/cam_to_cam.txt'
lidar2_cam6_file = './calibration/lidar2_to_cam6.txt'
lidar1_cam8_file = './calibration/lidar1_to_cam8.txt'

# create some utils funcion to read the calibration matrx 
def get_lidar_to_cam_extrinsic_matrix(file_path):
    # either lidar2_to_cam6.txt or lidar1_to_cam8.txt
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize the rotation matrix (R) and translation vector (T)
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))

    for line in lines:
        if line.startswith('R:'):
            values = line.strip().split()[1:]
            R = np.array(values, dtype=float).reshape(3, 3)
        elif line.startswith('T:'):
            values = line.strip().split()[1:]
            T = np.array(values, dtype=float).reshape(3, 1)
        # delta_f and delta_c are ignored based on the given information

    # Form the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T.flatten()
    return transformation_matrix

def get_single_camera_intrinsic(camera_number):
    # file_path = '../datasets/Safety_Challenge_data/validation_data/calibration/camera4_intrinsic.txt'
    file_path = cam_to_cam_file
    camera_key = f'K_VisualCamera{camera_number}'
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    intrinsic_matrix = None
    
    for line in lines:
        if line.startswith(camera_key):
            values = line.strip().split()[1:]
            intrinsic_matrix = np.array(values, dtype=float).reshape(3, 3)
            break
    if intrinsic_matrix is None:
        raise ValueError(f'Intrinsic matrix for {camera_key} not found in the file.')    
    return intrinsic_matrix
def get_single_camera_distortion(camera_number):
    file_path = cam_to_cam_file
    camera_key = f'D_VisualCamera{camera_number}'
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    distortion_matrix = None
    
    for line in lines:
        if line.startswith(camera_key):
            values = line.strip().split()[1:]
            distortion_matrix = np.array(values, dtype=float)
            break
    if distortion_matrix is None:
        raise ValueError(f'Distortion matrix for {camera_key} not found in the file.')
    return distortion_matrix

def get_camera_intrinsic(file_path, camera_number):

    file_path = cam_to_cam_file
    camera_key = f'K_VisualCamera{camera_number}'
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    intrinsic_matrix = None
    
    for line in lines:
        if line.startswith(camera_key):
            values = line.strip().split()[1:]
            intrinsic_matrix = np.array(values, dtype=float).reshape(3, 3)
            break
    
    if intrinsic_matrix is None:
        raise ValueError(f'Intrinsic matrix for {camera_key} not found in the file.')
    
    return intrinsic_matrix

def get_camera_transformation_matrix(file_path, from_camera_number, to_camera_number):
    camera_key = f'T_VisualCamera{from_camera_number}-VisualCamera{to_camera_number}'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    transformation_matrix = None
    for line in lines:
        if line.startswith(camera_key):
            values = line.strip().split()[1:]
            transformation_matrix= np.array(values, dtype=float).reshape(4, 4)
            break
    if transformation_matrix is None:
        raise ValueError(f'Transformation matrix for {camera_key} not found in the file.')
    return transformation_matrix

def get_lidar_transformation_matrix(file_path, from_lidar_number, to_lidar_number):
    lidar_key = f'T_Lidar{from_lidar_number}-Lidar{to_lidar_number}'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    transformation_matrix = None
    for line in lines:
        if line.startswith(lidar_key):
            values = line.strip().split()[1:]
            transformation_matrix= np.array(values, dtype=float).reshape(4, 4)
            break
    if transformation_matrix is None:
        raise ValueError(f'Transformation matrix for {lidar_key} not found in the file.')
    return transformation_matrix
## ---- below are some examples to get the transformation matrix by using the above functions ---- ##
## ---- start by defining your transform path ---- ##
def get_lidar2_camera1_projection_matrix():
    # get the projection matrix for the LiDAR to camera1
    # T_lidar2_camera1 = P_camera1 @ T_camera_5_camera_1 @ T_camera6_camera5 @ T_lidar2_camera_6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera6_camera5 = np.linalg.inv(get_camera_transformation_matrix(cam_to_cam_file, 5, 6)) # 4x4
    T_camera1_camera5 = get_camera_transformation_matrix(cam_to_cam_file, 1, 5) # 4x4
    # inverse of T_camera1_camera5 to get T_camera5_camera1
    T_camera5_camera1 = np.linalg.inv(T_camera1_camera5) # 4x4
    # get the intrinsic matrix for camera1
    K_camera1 = get_camera_intrinsic(cam_to_cam_file, 1) # 3x3
    # add a column of zeros to K_camera1 to get 3x4
    P_camera1 = np.hstack((K_camera1, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    T_lidar2_camera1 = T_camera5_camera1 @ T_camera6_camera5 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    P_lidar2_camera1 = P_camera1 @ T_camera5_camera1 @ T_camera6_camera5 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    return P_lidar2_camera1, T_lidar2_camera1

def get_lidar2_camera2_projection_matrix():
    # get the projection matrix for the LiDAR to camera2
    # T_lidar2_camera2 = P_camera2 @ T_camera7_camera2 @T_camera6_camera7 @ T_lidar2_camera6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera6_camera7 = get_camera_transformation_matrix(cam_to_cam_file, 6, 7) # 4x4
    T_camera2_camera7 = get_camera_transformation_matrix(cam_to_cam_file, 2, 7) # 4x4
    # inverse of T_camera2_camera7 to get T_camera7_camera2
    T_camera7_camera2 = np.linalg.inv(T_camera2_camera7) # 4x4
    # get the intrinsic matrix for camera2
    K_camera2 = get_camera_intrinsic(cam_to_cam_file, 2) # 3x3
    # add a column of zeros to K_camera2 to get 3x4
    P_camera2 = np.hstack((K_camera2, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    T_lidar2_camera2 = T_camera7_camera2 @ T_camera6_camera7 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    P_lidar2_camera2 = P_camera2 @ T_camera7_camera2 @ T_camera6_camera7 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    return P_lidar2_camera2, T_lidar2_camera2

def get_lidar2_camera3_projection_matrix():
    # get the projection matrix for the LiDAR to camera3
    # T_lidar2_camera3 = P_camera3 @ T_camera5_camera3 @ T_camera6_camera5 @ T_lidar2_camera6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera6_camera5 = np.linalg.inv(get_camera_transformation_matrix(cam_to_cam_file, 5, 6)) # 4x4
    T_camera5_camera3 = np.linalg.inv(get_camera_transformation_matrix(cam_to_cam_file, 3, 5)) # 4x4
    # intrinsic matrix for camera3
    K_camera3 = get_camera_intrinsic(cam_to_cam_file, 3) # 3x3
    # add a column of zeros to K_camera3 to get 3x4
    P_camera3 = np.hstack((K_camera3, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    T_lidar2_camera3 = T_camera5_camera3 @ T_camera6_camera5 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    P_lidar2_camera3 = P_camera3 @ T_camera5_camera3 @ T_camera6_camera5 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    return P_lidar2_camera3, T_lidar2_camera3

def get_lidar2_camera4_projection_matrix():
    # get the projection matrix for the LiDAR to camera4
    # T_lidar2_camera4 = P_camera4 @ T_camera6_camera4 @ T_lidar2_camera_6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera4_camera6 = get_camera_transformation_matrix(cam_to_cam_file, 4, 6) # 4x4
    # inverse of T_camera4_camera6 to get T_camera6_camera4
    T_camera6_camera4 = np.linalg.inv(T_camera4_camera6) # 4x4
    # get the intrinsic matrix for camera4
    K_camera4 = get_camera_intrinsic(cam_to_cam_file, 4) # 3x3
    # add a column of zeros to K_camera4 to get 3x4
    P_camera4 = np.hstack((K_camera4, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    T_lidar2_camera4 = T_camera6_camera4 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 = 3x4
    P_lidar2_camera4 = P_camera4 @ T_camera6_camera4 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 = 3x4
    return P_lidar2_camera4, T_lidar2_camera4

def get_lidar2_camera5_projection_matrix():
    # get T_lidar2_camera5 = P_camera5 @ T_camera6_camera5 @ T_lidar2_camera6
    # P_lidar2_camera4, T_lidar2_camera4 = get_lidar2_camera4_projection_matrix()
    # add the transformation from GT to lidar2
    # T_GT_lidar2 = np.array([[0,1,0, 0],
    #                        [-1,0,0, 0],
    #                        [0,0,1, 0],
    #                        [0, 0, 0, 1]])
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera5_camera6 = get_camera_transformation_matrix(cam_to_cam_file, 5, 6) # 4x4
    # inverse of T_camera5_camera6 to get T_camera6_camera5
    T_camera6_camera5 = np.linalg.inv(T_camera5_camera6) # 4x4
    P_camera5 = get_camera_intrinsic(cam_to_cam_file, 5) # 3x3
    P_camera5 = np.hstack((P_camera5, np.zeros((3, 1)))) # 3x4
    
    # T_lidar2_camera5 = T_camera4_camera5 @ T_lidar2_camera4 # 4x4 @ 4x4 @ 4x4 = 4x4 # without the projection matrix
    # P_lidar2_camera5 = P_camera5 @ T_camera4_camera5 @ T_lidar2_camera4 # 3x4 @ 4x4 @ 4x4 = 3x4
    T_lidar2_camera5 = T_camera6_camera5@ T_lidar2_camera6 # 4x4 @ 4x4 @ 4x4 = 4x4
    P_lidar2_camera5 = P_camera5 @ T_camera6_camera5@ T_lidar2_camera6# 3x4 @ 4x4 @ 4x4 @ 4x4 =
    return P_lidar2_camera5, T_lidar2_camera5

def get_lidar2_camera6_projection_matrix():
    # get the projection matrix          the LiDAR to camera6
    # T_lidar2_camera6 = P_camera6 @ @ T_lidar2_camera_6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    # get the intrinsic matrix for camera6
    K_camera6 = get_camera_intrinsic(cam_to_cam_file, 6) # 3x3
    # add a column of zeros to K_camera6 to get 3x4
    P_camera6 = np.hstack((K_camera6, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    P_lidar2_camera6 = P_camera6 @ T_lidar2_camera6 # 3x4 @ 4x4 = 3x4
    return P_lidar2_camera6, T_lidar2_camera6

def get_lidar2_camera7_projection_matrix():
    # get the projection matrix for the LiDAR to camera7
    # T_lidar2_camera7 = P_camera7 @ T_camera6_camera7 @ T_lidar2_camera_6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera6_camera7 = get_camera_transformation_matrix(cam_to_cam_file, 6, 7) # 4x4
    # get the intrinsic matrix for camera7
    K_camera7 = get_camera_intrinsic(cam_to_cam_file, 7) # 3x3
    # add a column of zeros to K_camera7 to get 3x4
    P_camera7 = np.hstack((K_camera7, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    T_lidar2_camera7 = T_camera6_camera7 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 = 3x4
    P_lidar2_camera7 = P_camera7 @ T_camera6_camera7 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 = 3x4
    return P_lidar2_camera7, T_lidar2_camera7

def get_lidar2_camera8_projection_matrix():
    # get the projection matrix for the LiDAR to camera8
    # T_lidar2_camera8 = P_camera8 @ T_camera4_camera8 @ T_camera6_camera4 @ T_lidar2_camera_6
    T_lidar2_camera6 = get_lidar_to_cam_extrinsic_matrix(lidar2_cam6_file) # 4x4
    T_camera6_camera4 = np.linalg.inv(get_camera_transformation_matrix(cam_to_cam_file, 4, 6)) # 4x4
    T_camera4_camera8 = np.linalg.inv(get_camera_transformation_matrix(cam_to_cam_file, 8, 4)) # 4x4

    # get the intrinsic matrix for camera8
    K_camera8 = get_camera_intrinsic(cam_to_cam_file, 8) # 3x3
    # add a column of zeros to K_camera8 to get 3x4
    P_camera8 = np.hstack((K_camera8, np.zeros((3, 1)))) # 3x4
    # get the projection matrix with the path
    T_lidar2_camera8 = T_camera4_camera8 @ T_camera6_camera4 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    P_lidar2_camera8 = P_camera8 @ T_camera4_camera8 @ T_camera6_camera4 @ T_lidar2_camera6 # 3x4 @ 4x4 @ 4x4 @ 4x4 = 3x4
    return P_lidar2_camera8, T_lidar2_camera8
def get_lidar1_camera4_projection_matrix():
    # lidar1 to lidar2
    T_lidar1_lidar2 = get_lidar_transformation_matrix(cam_to_cam_file, 1, 2)
    # lidar2 to camera4
    P_lidar2_camera4, T_lidar2_camera4 = get_lidar2_camera4_projection_matrix()
    # lidar1 to camera4
    T_lidar1_camera4 = P_lidar2_camera4 @ T_lidar1_lidar2
    return T_lidar1_camera4, None

def draw_2d_bbox(image, corner_2d):
    # draw the 2D bounding box on the image
    # image: the image to draw the bounding box
    # corner_2d: 8x2, the 2D corner of the bounding box
    # return: the image with the bounding box
      # Define the pairs of points to draw the lines between
    pairs = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Top face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Bottom face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
    ]
    corner_2d = corner_2d.astype(int)

    # Draw the bounding box
    for i, j in pairs:
        cv.line(image, tuple(corner_2d[i]), tuple(corner_2d[j]), (0, 255, 0), 2)

    return image

# draw the flattened 3D bounding box
def draw_2d_bbox_flattern(image, frame_num, flattern_corner_2d, obj_name):
    min_x, min_y, max_x, max_y = flattern_corner_2d
    cv.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)
    # add the object name
    cv.putText(image, obj_name, (min_x, min_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    # put text on the right coder of the image, showing the frame number
    cv.putText(image, f'Frame: {frame_num}', (image.shape[0]- 20, image.shape[1]-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return image

# 
def flattern_2d_bbox(corner_2d):
    corner_2d = corner_2d.astype(int)
    min_x = np.min(corner_2d[:, 0])
    max_x = np.max(corner_2d[:, 0])
    min_y = np.min(corner_2d[:, 1])
    max_y = np.max(corner_2d[:, 1])
    flattern_2d = np.array([min_x, min_y, max_x, max_y])
    return flattern_2d




if __name__ == '__main__':
    # some test code
    # P_lidar2_camera4, T_lidar2_camera4 = get_lidar2_camera4_projection_matrix()
    # P_lidar2_camera6, T_lidar2_camera6 = get_lidar2_camera6_projection_matrix()
    # P_lidar2_camera5, T_lidar2_camera5 = get_lidar2_camera5_projection_matrix()
    # P_lidar2_camera1, T_lidar2_camera1 = get_lidar2_camera1_projection_matrix()
    # P_lidar2_camera2, T_lidar2_camera2 = get_lidar2_camera2_projection_matrix()
    # P_lidar2_camera3, T_lidar2_camera3 = get_lidar2_camera3_projection_matrix()
    # P_lidar2_camera7, T_lidar2_camera7 = get_lidar2_camera7_projection_matrix()
    # P_lidar2_camera8, T_lidar2_camera8 = get_lidar2_camera8_projection_matrix()
    # print('P_lidar2_camera1:\n', P_lidar2_camera1)
    # print('P_lidar2_camera2:\n', P_lidar2_camera2)
    # print('P_lidar2_camera3:\n', P_lidar2_camera3)
    # print('P_lidar2_camera4:\n', P_lidar2_camera4)
    # # print('T_lidar2_camera4:\n', T_lidar2_camera4)
    # print('P_lidar2_camera6:\n', P_lidar2_camera6)
    # # print('T_lidar2_camera6:\n', T_lidar2_camera6)
    # print('P_lidar2_camera5:\n', P_lidar2_camera5)
    # # print('T_lidar2_camera5:\n', T_lidar2_camera5)
    # print('P_lidar2_camera7:\n', P_lidar2_camera7)
    # print('P_lidar2_camera8:\n', P_lidar2_camera8)

    load_conventional_pipeline_result(Run_num= 55)