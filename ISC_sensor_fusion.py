# ===================================================================================
# Copyright (C) 2024 @Haishan Liu, hliu240@ucr.edu
#
# This file contains the key helper functions to fuse the camera and LiDAR data.
# ===================================================================================

import numpy as np
import cv2
from scipy.spatial import ConvexHull
from sklearn import linear_model
from ISC_utils import *
from mmdet.apis import DetInferencer
import mmdet
from sklearn.decomposition import PCA
import torch
import math

import matplotlib.pyplot as plt
import os


CLUSTER_LABELS = {i: object for i, object in enumerate(mmdet.evaluation.functional.coco_classes())}
# 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
TAGRET_LABELS = set([0, 1, 2, 3, 5, 7])


def get_fused_detection(image, bin_path, model, T_velo_cam, T_cam_velo,K , D, draw_boxes=True, draw_depth=True):
    ''' Obtains the detections from the camera and associates them with LiDAR points.
       Get the lidar points that are within the bounding box of the detected objects.
       then calculate the center of the points that are within the bounding box.
        Inputs:
            image - rgb image to run detection on
            bin_path - path to LiDAR bin file
            T_velo_cam - transformation from LiDAR to camera## (u,v,z) space
            model - detection model (this functions assumes a yolo5 model)
                  - any detector can be used as long as it has the following attributes:
                    show, xyxy
        Output:
            bboxes - array of detected bounding boxes
            labels - detected classes
            scores - confidence scores
            velo_uv - LiDAR points porjected to camera uvz coordinate frame
            centers - centers of the detected objects in LiDAR xyz coordinates
        '''
    ## 1. compute detections in the left image
    detections = model(image, pred_score_thr=0.1, show=False, no_save_pred=True, return_vis=False)
    for i, result in enumerate(detections['predictions']):
        labels = np.array(result['labels'])
        scores = np.array(result['scores'])
        bboxes = np.array(result['bboxes']).astype(int) # min_x, min_y, max_x, max_y

        # fileter out the labels that are not interested
        mask = (scores >= 0.3) & (labels != 9) & (labels != 4)
        labels, scores, bboxes = labels[mask], scores[mask], bboxes[mask]
        
        # only keep the interested labels
        mask2 = np.isin(labels, list(TAGRET_LABELS))
        labels, scores, bboxes = labels[mask2], scores[mask2], bboxes[mask2]

        # set confidence threshold for car to be >= 0.5, for person to be >= 0.3
         # Define class-wise confidence thresholds
        class_thresholds = {
            0: 0.3,  # person
            1: 0.3,  # bicycle
            2: 0.5,  # car
            3: 0.3,  # motorcycle
            5: 0.5,  # bus
            7: 0.5,  # truck
            # Add other class labels and their thresholds as needed
        }
        
        # Create an array of thresholds corresponding to each detection
        thresholds = np.array([class_thresholds[label] for label in labels])

        # Create a mask where each detection meets its class-specific threshold
        mask3 = scores >= thresholds
        labels, scores, bboxes = labels[mask3], scores[mask3], bboxes[mask3]


    
    # step 1: undistort the image
    undistorted_image = cv2.undistort(image, K, D)
    # step 2: project velo --> point_uvz, mainly use this function for syncronization verification
    velo_uvz = project_velobin2uvz(bin_path, T_velo_cam, undistorted_image, remove_plane=False)


    # step 3: get the point_uvz centers for the detected objects
    bboxes, scores, labels, clusters, centers = associate_cam_to_lidar(bboxes= bboxes, labels= labels, scores= scores, velo_uvz=velo_uvz, T_cam_velo= T_cam_velo)

    # draw boxes on image
    if draw_boxes:
        for i, bbox in enumerate(bboxes):
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            cv2.rectangle(image, pt1, pt2, (0, 255, 0))
            text_position = (pt1[0], pt1[1] - 10)
        
            # Add the bounding box number (index) to the image
            cv2.putText(image, f'#{i}-{CLUSTER_LABELS[labels[i]]}-{scores[i]:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)  # Red text
# # get uvz centers for detected objects
    # bboxes = get_uvz_centers(image, 
    #                          velo_uvz, 
    #                          bboxes, 
    #                          draw=draw_depth)
    
    return bboxes, labels, scores, clusters, centers, velo_uvz

def get_fused_detection_multi_methods(cap, image_num, bin_path, model, T_velo_cam, T_cam_velo,K , D, draw_boxes=True, draw_depth=True):
    ''' This is the interface from Chuheng's new results. But I will explore later. 


        Obtains the detections from the camera and associates them with LiDAR points.
       Get the lidar points that are within the bounding box of the detected objects.
       then calculate the center of the points that are within the bounding box.
        Inputs:
            cap - video capture object
            image_num - this is the frame number of the image
            bin_path - path to LiDAR bin file
            T_velo_cam - transformation from LiDAR to camera## (u,v,z) space
            model - detection model (this functions assumes a yolo5 model)
                  - any detector can be used as long as it has the following attributes:
                    show, xyxy
        Output:
            bboxes - array of detected bounding boxes
            labels - detected classes
            scores - confidence scores
            velo_uv - LiDAR points porjected to camera uvz coordinate frame
            centers - centers of the detected objects in LiDAR xyz coordinates
        '''
    ## 1. compute detections in the left image
    detections = model(image, pred_score_thr=0.1, show=False, no_save_pred=True, return_vis=False)
    for i, result in enumerate(detections['predictions']):
        labels = np.array(result['labels'])
        scores = np.array(result['scores'])
        bboxes = np.array(result['bboxes']).astype(int) # min_x, min_y, max_x, max_y

        # fileter out the labels that are not interested
        mask = (scores >= 0.3) & (labels != 9) & (labels != 4)
        labels, scores, bboxes = labels[mask], scores[mask], bboxes[mask]
        
        # only keep the interested labels
        mask2 = np.isin(labels, list(TAGRET_LABELS))
        labels, scores, bboxes = labels[mask2], scores[mask2], bboxes[mask2]

        # set confidence threshold for car to be >= 0.5, for person to be >= 0.3
         # Define class-wise confidence thresholds
        class_thresholds = {
            0: 0.3,  # person
            1: 0.3,  # bicycle
            2: 0.5,  # car
            3: 0.3,  # motorcycle
            5: 0.5,  # bus
            7: 0.5,  # truck
            # Add other class labels and their thresholds as needed
        }
        
        # Create an array of thresholds corresponding to each detection
        thresholds = np.array([class_thresholds[label] for label in labels])

        # Create a mask where each detection meets its class-specific threshold
        mask3 = scores >= thresholds
        labels, scores, bboxes = labels[mask3], scores[mask3], bboxes[mask3]


    
    # step 1: undistort the image
    undistorted_image = cv2.undistort(image, K, D)
    # step 2: project velo --> point_uvz, mainly use this function for syncronization verification
    velo_uvz = project_velobin2uvz(bin_path, T_velo_cam, undistorted_image, remove_plane=False)


    # step 3: get the point_uvz centers for the detected objects
    bboxes, scores, labels, clusters, centers = associate_cam_to_lidar(bboxes= bboxes, labels= labels, scores= scores, velo_uvz=velo_uvz, T_cam_velo= T_cam_velo)

    # draw boxes on image
    if draw_boxes:
        for i, bbox in enumerate(bboxes):
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            cv2.rectangle(image, pt1, pt2, (0, 255, 0))
            text_position = (pt1[0], pt1[1] - 10)
        
            # Add the bounding box number (index) to the image
            cv2.putText(image, f'#{i}-{CLUSTER_LABELS[labels[i]]}-{scores[i]:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)  # Red text
# # get uvz centers for detected objects
    # bboxes = get_uvz_centers(image, 
    #                          velo_uvz, 
    #                          bboxes, 
    #                          draw=draw_depth)
    
    return bboxes, labels, scores, clusters, centers, velo_uvz


# ============================================================================================
# pipeline functions 
def project_velobin2uvz(bin_path, T_velo_cam, image, remove_plane=True):
    ''' Projects LiDAR point cloud onto the image coordinate frame (u, v, z)
        Inputs: bin_path - path to LiDAR bin file
        T_uvz_velo - 3x4 transformation matrix from LiDAR to camera reference frame
        image - rgb image to project the LiDAR points onto
        '''
    # T_uvz_velo: 3x4 transformation matrix from LiDAR to camera reference frame

    # get homogeneous LiDAR points from bin file, load the points   
    xyzw = bin2xyzw(bin_path, remove_plane)
    # xyzw: 4xN array of LiDAR points
   

    # project velo (x, z, y, w) onto camera (u, v, z) coordinates
    # change this function to yanyu's function
    velo_uvz = xyzw2uvz(xyzw, T_velo_cam, image, remove_outliers=True)
    # velo_uvz: 3xN array of LiDAR points in camera (u, v, z) coordinates
    
    return velo_uvz

# ============================================================================================
# pipeline functions 

def get_cluster_centers_and_rotations(clusters, T_cam_velo):
    ''''clusters is a list of np.array from the associate_cam_to_lidar function, in uvz coordinates
    each array is the points that are associated with one bbox. 
    clusters[i][:3], i.e. the first three columns are the uvz coordinates of the points
    clusters[i][3] is the label of the bbox & clusters[i]
    clusters[i][4] is the score of the bbox & clusters[i]
    We calculate the convex hull in the xyz space
    input: clusters - list of np.array of points associated with each bbox
           T_cam_velo - transformation matrix from camera to LiDAR reference frame
    output: centers - list of centers of the detected objects in LiDAR xyz coordinates
    '''
    # case 1: if the cluster has more than 3 points, calculate the convex hull
    # case 2: if the cluster has 2 points, calculate the center of the line
    # case 3: if the cluster has 1 point, use the point as the center
    # case 4: if the cluster has 0 points, use the center of the bbox as the center, and the depth as the average of the bbox depth
    
    centers = [[] for _ in range(len(clusters))]
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            print(f'\t No lidar points associated with bbox {i}')
            continue
        # required: uvz is a Nx3 array
        # print(f'cluster: {cluster}')
        cluster_xyz = uvz2xyz(uvz= np.hstack((cluster[:,:3], np.ones((len(cluster),1)))), T_cam_velo= T_cam_velo)
        center_label = cluster[0, 3]
        center_score = cluster[0, 4]
        if center_label in [2, 5, 7]: # 2: car, 5: bus, 7: truck
            # vehicle should have at least 10 points to calculate the convex hull
            if len(cluster_xyz) > 20:
                hull_vertice = ConvexHull(cluster_xyz, qhull_options='QJ').vertices
                hull_pcd = cluster_xyz[hull_vertice]
                rotation = get_cluster_rotation(hull_pcd)
                # rotation_ziyan = compute_principal_directions(hull_pcd)
                center = np.mean(hull_pcd, axis=0)
                center = np.append(center, [center_label, center_score, rotation])
            else:
                center = []
                print(f'\t {len(cluster_xyz)} points associated with bbox {i}')
                continue
        else:
            if len(cluster_xyz) >= 4:
                hull_vertice = ConvexHull(cluster_xyz, qhull_options='QJ').vertices
                hull_pcd = cluster_xyz[hull_vertice]
                rotation = get_cluster_rotation(hull_pcd)
                # rotation_ziyan = compute_principal_directions(hull_pcd)
                center = np.mean(hull_pcd, axis=0)
                center = np.append(center, [center_label, center_score, rotation])
                # get the rotation of the convex hull
                print(f'\t {len(cluster_xyz)} points associated with bbox {i}')
        
            elif len(cluster_xyz) == 3:
                rotation = get_cluster_rotation(cluster_xyz)
                # rotation_ziyan = compute_principal_directions(cluster_xyz)
                center = np.mean(cluster_xyz, axis=0)
                center = np.append(center, [center_label, center_score, rotation])
                print(f'\t 3 points associated with bbox {i}')

            else:
                center = []
                # center = np.array([bbox[i, 0] + bbox[i, 2] / 2, bbox[i, 1] + bbox[i, 3] / 2, np.mean(velo_uvz[:, 2])])
                print(f'\t 1-2 points associated with bbox {i}')
        centers[i] = center
    # print(f'centers: {centers}')
    return centers

# ============================================================================================
# pipeline functions -- 
def get_cluster_rotation(cluster):
    '''cluster is a np.array from the convex hull function
    cluster[i][:3], i.e. the first three columns are the uvz coordinates of the points
    cluster[i][3] is the label of the bbox & clusters[i]
    cluster[i][4] is the score of the bbox & clusters[i]
    We calculate the convex hull in the xyz space
    input: cluster - list of np.array of points associated with each bbox
           
    output: rotation - list of centers of the detected objects in LiDAR xyz coordinates
    '''
    xy_points = cluster[:, :2]
    pca = PCA(n_components=2)
    pca.fit(xy_points)
    principal_axis = pca.components_[0]
    raw = np.arctan2(principal_axis[1], principal_axis[0]) -np.pi/2
    raw = limit_period(raw, 0, np.pi*2)
    # in radians
    return raw

# ============================================================================================
# pipeline functions ziyan's function to calculate the rotation

def limit_period(val,
                 offset: float = 0,
                 period: float = 2*np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    
    limited_val = val - np.floor(val / period + offset) * period
    limited_val = math.degrees(limited_val)
    return limited_val

def compute_principal_directions(points):
    # Compute the covariance matrix
    cov_matrix = np.cov(points, rowvar=False)
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # The eigenvectors correspond to the principal directions
    directions = eigenvectors[:,:2]
    combined_direction = np.mean(directions, axis=1)
    combined_direction = combined_direction / np.linalg.norm(combined_direction)  # Normalize the vector

    # Calculate raw
    yaw = math.atan2(combined_direction[1], combined_direction[0])
    yaw = limit_period(yaw, period=np.pi * 2)
    yaw = math.degrees(yaw)

    return yaw  



from torchvision.ops import nms

def reduce_redundant_detection(bboxes, scores, labels, threshold=0.5):
    '''Reduce redundant detection by applying non-maximum suppression
    input: bboxes - N x 4 array of bounding boxes
           scores - N x 1 array of confidence scores
           threshold - IoU threshold for non-maximum suppression
    output: reduced_bboxes - N x 4 array of reduced bounding boxes
            reduced_scores - N x 1 array of reduced confidence scores
    '''
    # convert bboxes to xyxy format
    keep = nms(torch.tensor(bboxes).float(), torch.tensor(scores).float(), threshold)
    reduced_bboxes = bboxes[keep]
    reduced_scores = scores[keep]
    reduced_labels = labels[keep]
    return reduced_bboxes, reduced_scores, reduced_labels

# ============================================================================================
# pipeline functions 
def associate_cam_to_lidar(bboxes, labels, scores, velo_uvz, T_cam_velo):
    '''associates to lidar points to the detected objects'''
    # before velo_uvz: 3 * N  
    velo_uvz = velo_uvz.T
    # after velo_uvz: N * 3

    points_2d = velo_uvz[:, :2]
    threshold_distance = 30
    # set confidence threshold for car to be >= 0.5, for person to be >= 0.3

    # First conduct NMS to reduce redundant detections
    reduced_bboxes, reduced_scores, reduced_labels = reduce_redundant_detection(bboxes, scores, labels, threshold=0.2)
    bboxes, scores, labels = reduced_bboxes, reduced_scores, reduced_labels

    # Ensure bboxes is a 2D array
    bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)

    # Ensure scores and labels are 1D arrays
    scores = np.array(scores).flatten()
    labels = np.array(labels).flatten()

    # Handle scalar scores and labels
    if np.isscalar(scores):
        scores = np.array([scores])
    if np.isscalar(labels):
        labels = np.array([labels])

    # Ensure arrays have the same length
    assert scores.shape[0] == labels.shape[0] == bboxes.shape[0], "Arrays must have the same length."

    # Sort by labels ascending and scores descending
    sorted_idx = np.lexsort((-scores, labels))

    print('sorted_idx', sorted_idx)
    print('scores', scores)

    # Apply sorted indices
    bboxes = bboxes[sorted_idx]
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    
    # for each bbox, get the points that are within the threshold distance
    clusters = [[]for _ in range(len(bboxes))]
    for i, bbox in enumerate(bboxes):
        contour =  np.array([[bbox[0], bbox[1]], # top left
                            [bbox[2], bbox[1]], # top right
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]]])
        distances = np.array([cv2.pointPolygonTest(contour, (point[0], point[1]), measureDist=True)
                             for point in points_2d])
   
        # ===== Associate points to the bbox if within the threshold distance ===== #
       
        inside_or_near = np.where(distances >= -threshold_distance)[0]
        # print(f'inside_or_near: {inside_or_near}')
        points_inside = velo_uvz[inside_or_near] # Nx3 array of points that are within the bbox
        
        # with the sorted bbox, we can remove the points that are within the bbox, this is to avoid the same points are assigned to multiple bboxes
        # withtout replacement, remove the points that are within the bbox
        # remove the points_inside from the velo_uvz
        velo_uvz = np.delete(velo_uvz, inside_or_near, axis=0)
        points_2d = velo_uvz[:, :2]

        # add the label to the points
        cluster_label_col = np.ones((len(points_inside), 1)) * labels[i]
        cluster_score_col = np.ones((len(points_inside), 1)) * scores[i]
        points_inside = np.hstack((points_inside, cluster_label_col, cluster_score_col))
        # clusters[i].extend(velo_uvz[inside_or_near])
        '''point_inside: Nx5 array of points that are within the bbox, [x, y, z, lable, score]the last column is the label'''
        clusters[i].extend(points_inside)
    # velo_uvz may have some points that are not associated with any bbox:
    # reason: the points are too far away from the bbox
    # reason: false negative from the detr detection model, need to relied on the histroy information
    
    # 每一行是一个bbox对应的lidar点
    clusters = [np.array(cluster) for cluster in clusters]
    centers = get_cluster_centers_and_rotations(clusters, T_cam_velo)
    
    return bboxes, scores, labels, clusters, centers

# ============================================================================================
# file access functions
def bin2xyzw(bin_path, remove_plane=False):
    ''' Reads LiDAR bin file and returns homogeneous (x,y,z,1) LiDAR points
    input bin_path - path to LiDAR bin file
    output xyzw - 4xN array of LiDAR points'''
    # read in LiDAR data
    scan_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1,4))

    #  xyz: Nx3 array of LiDAR points
    xyz = scan_data[:, 0:3] 
  

    # delete negative liDAR points
    # xyz = np.delete(xyz, np.where(xyz[3, :] < 0), axis=1)
    R_label_to_velo = np.array([[0,1,0],
                           [-1,0,0],
                           [0,0,1]])
    
    xyz = R_label_to_velo @ xyz.T
    # xyz = xyz.T
    # xyz: 3xN array of LiDAR points

    # print(xyz)
    # use ransac to remove ground plane
    if remove_plane:
        ransac = linear_model.RANSACRegressor(
                                      linear_model.LinearRegression(),
                                      residual_threshold=0.1,
                                      max_trials=5000
                                      )

        X = xyz[:, :2]
        y = xyz[:, -1]
        ransac.fit(X, y)
        
        # remove outlier points (i.e. remove ground plane)
        mask = ransac.inlier_mask_
        xyz = xyz[~mask]

    # conver to homogeneous LiDAR points
    # print(xyz.shape)
    # insert 1 at the third row to make it homogeneous
    # xyz: 3xN array of LiDAR points
    # xyzw = np.vstack((xyz, np.ones((1, len(xyz)))))
    xyzw = np.insert(xyz, 3, 1, axis=0)
    # print('after read in:', xyzw.shape)
  
    return xyzw

# ============================================================================================
# transformation functions
def xyzw2uvz(xyzw, T_velo_cam, image=None, remove_outliers=True):
    ''' maps xyxw homogeneous points to camera (u,v,z) space. The xyzw points can 
        either be velo/LiDAR or GPS/IMU, the difference will be marked by the 
        transformation matrix T. To which camera, the difference will be marked by T
        core: xyzw --- > uvz

        Inputs: xyzw (4xN) array of xyz points
                T_velo_cam (3x4): transformation matrix from LiDAR to camera reference frame
        Outputs: camera (3xN) array of camera coordinates (u,v,z)
        '''
    # convert to (left) camera coordinates
    # print('before project lidar',xyz.shape)
    # print('T',T.shape)
    velo_uvz =  T_velo_cam @ xyzw # 3x4 @ 4xN = 3xN
    
    # delete negative camera points
    velo_uvz  = np.delete(velo_uvz , np.where(velo_uvz [2,:] < 0)[0], axis=1) 

    # get camera coordinates u,v,z
    velo_uvz[:2] /= velo_uvz[2, :]

    # remove outliers (points outside of the image frame)
    if remove_outliers:
        u, v, z = velo_uvz
        img_h, img_w, _ = image.shape
        u_out = np.logical_or(u < 0, u > img_w)
        v_out = np.logical_or(v < 0, v > img_h)
        outlier = np.logical_or(u_out, v_out)
        velo_uvz = np.delete(velo_uvz, np.where(outlier), axis=1)

    return velo_uvz

def uvz2xyz(uvz, T_cam_velo):
    ''' Transforms the uvz coordinates to xyz coordinates. The xyz coordinate
        frame is specified by the transformation matrix T. The transformation
        may include:
            uvz -> xyz (LiDAR)
            uvz -> xyz (IMU)

        Inputs: uvz (Nx3) array of uvz coordinates
        Outputs: xyz (Nx3) array of xyz coordinates
        '''
    # covnert to homogeneous representation
    uvzw = np.hstack((uvz[:, :2] * uvz[:, 2][:, None], 
                      uvz[:, 2][:, None],
                      np.ones((len(uvz[:, :3]), 1))))
    
    # perform homogeneous transformation
    xyzw = T_cam_velo @ uvzw.T
    
    # get xyz coordinates
    xyz = xyzw[:3, :].T

    return xyz
# ============================================================================================



# ============================================================================================
# visualization functions
from matplotlib import cm

# get color map function
rainbow_r = cm.get_cmap('rainbow_r', lut=100)
get_color = lambda z : [255*val for val in rainbow_r(int(z.round()))[:3]]

def draw_velo_on_image(velo_uvz, image, color_map=get_color):
   
    # unpack LiDAR points
    u, v, z = velo_uvz

    # draw LiDAR point cloud on blank image
    for i in range(len(u)):
        cv2.circle(image, (int(u[i]), int(v[i])), 5, 
                   color_map(z[i]), -1);

    return image
# Define the get_color function
def get_color(cluster_label):
     # Manually define a set of distinct colors if necessary
    distinct_colors = {
        0: (255, 0, 0),  # Red for 'person'
        2: (0, 255, 0),  # Green for 'car'
        # Add more specific mappings if needed
    }
    
    # Check if the label has a predefined distinct color
    if cluster_label in distinct_colors:
        return distinct_colors[cluster_label]
    # Use a colormap (e.g., 'tab20' for up to 20 distinct colors)
    colormap = plt.get_cmap('tab20')

    # Normalize the cluster label to a value between 0 and 1
    max_label = len(CLUSTER_LABELS) - 1  # Number of classes
    normalized_label = cluster_label / max_label

    # Get the RGBA color from the colormap and convert to BGR for OpenCV
    color = colormap(normalized_label)
    
    # Convert RGBA to BGR format and scale to 0-255
    bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
    
    return bgr_color

# Function to draw the legend on the image
def draw_legend(image, labels, color_map=get_color, position=(50, 50), box_size=20, text_offset=30, font_scale=1.5):
    y_offset = position[1]
    for label in labels:
        color = color_map(label)
        # Draw the color box
        cv2.rectangle(image, (position[0], y_offset), (position[0] + box_size, y_offset + box_size), color, -1)
        # Draw the text label
        cv2.putText(image, CLUSTER_LABELS[label], (position[0] + text_offset, y_offset + box_size - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0,255), thickness=2, lineType =cv2.LINE_AA)
        y_offset += box_size + 10

def draw_cluster_on_image(velo_uvz, image, color_map=get_color):
    print('velo_uvz', velo_uvz.shape)
    # unpack LiDAR points
    u, v, z, labels, scores = velo_uvz
    # draw LiDAR point cloud on blank image
    for i in range(len(u)):
        cv2.circle(image, (int(u[i]), int(v[i])), 5, 
                   color_map(labels[i]), -1); 

    # get the unique labels
    unique_labels = np.unique(labels)
    # draw the legend
    # print('unique labels', unique_labels)
    draw_legend(image, unique_labels, color_map)   
    return image

def concatenate_images(image_list, layout, target_size=(1920, 1080)):
    """
    Concatenate images according to the specified layout and resize images to target size.
    
    Parameters:
    - image_list: List of images to concatenate.
    - layout: Tuple specifying the number of images in each row, e.g., (3, 2).
    - target_size: Tuple specifying the width and height to resize images, e.g., (1920, 1080).
    
    Returns:
    - Concatenated image.
    """
    # Initialize an empty list to hold the rows of images
    rows = []
    
    # Starting index for images in the list
    idx = 0

    # image_list[-1] = cv2.resize(image_list[-1], target_size)
    # assert image_list[-1].shape[0] == target_size[1] and image_list[-1].shape[1] == target_size[0]
    one_row_num = layout[0]
    target_y_size = target_size[0] * one_row_num
    # print('target_y_size', target_y_size)
    # print('image_list[-1].shape', image_list[-1].shape)
    
    for num_images in layout:
        # Extract the corresponding images for this row
        row_images = image_list[idx:idx + num_images]
        idx += num_images
        
        # Resize each image to the target size
        row_images = [cv2.resize(img, target_size) for img in row_images]
        
        # Concatenate the images horizontally to form a row
        row = np.hstack(row_images)
        
        # Add the row to the list of rows
        # print('row.shape', row.shape)
        rows.append(row)
    
    # Concatenate all rows vertically to form the final image
    for i, row in enumerate(rows):
        if row.shape[1] < target_y_size:
            row = np.hstack((row,  np.ones((row.shape[0], target_y_size - row.shape[1], 3)))) 
            rows[i] = row
            # print('append row.shape', row.shape) 

    big_image = np.vstack(rows)
    
    return big_image

#================================================
# miscellaneous functions

def center_empty_filter(clusters: list) -> np.ndarray:
    # Filter out empty arrays (arrays with zero elements)
    '''centers: a list of arrays'''
    # check
    clusters= [cluster for cluster in clusters if len(cluster) > 0]

    # Concatenate the non-empty arrays
    if clusters:  # Ensure there is something to concatenate
        big_array = np.concatenate(clusters, axis=0)
    else:
        big_array = np.array([])  # If all arrays were empty, return an empty array

    return big_array

def write_out_fused_result(run_fused_result, fusion_label_dir, Run_num):
    '''Write out the fused result to a txt file'''
    '''bin_idx.pid, object_1, 
       bin_idx.pid, object_2,'''
    with open(f'{fusion_label_dir}/Run_{Run_num}_fused_result.txt', 'w') as f:
        for bin_idx, proposals in run_fused_result.items():
            # when no proposals, skip
            if len(proposals) == 0:
                continue
            for i, proposal in enumerate(proposals):
                # bin_index label, x, y, z, score, rotation, rotation_ziyan
                f.write(f'{bin_idx:06d}.pid, {CLUSTER_LABELS[proposal[3]]}, {proposal[0]}, {proposal[1]}, {proposal[2]}, {proposal[4]}, {proposal[5]}\n')
    
def suppress_closed_center(center, centers,threshold=0.2):
    '''Suppress the closed centers'''
    # center: 1x7 array
    # centers: Nx7 array
    # threshold: the distance threshold
    # return the suppressed center
    if len(centers) == 0:
        return center
    distances = np.linalg.norm(centers[:, :3] - center[:3], axis=1)
    if np.any((distances < threshold) & (distances > 0)):
        return np.array([])
    return center

def per_class_suppression(proposals, threshold=0.2):
    '''Suppress the closed centers for each class'''
    # proposals: Nx7 array
    suppressed_detections  = []
    while proposals.shape[0] > 0:
        max_score_idx = np.argmax(proposals[:, 4])
        max_score_proposal = proposals[max_score_idx]
        
        # Add the max score proposal to the list of suppressed detections
        suppressed_detections.append(max_score_proposal)
        
        # Delete the max score proposal from the proposals list
        proposals = np.delete(proposals, max_score_idx, axis=0)
        
        # Iterate over remaining proposals and filter based on distance
        proposals_to_keep = []
        for detection in proposals:
            distance = np.linalg.norm(detection[:3] - max_score_proposal[:3])
            if distance >= 0.3:
                proposals_to_keep.append(detection)   
         # Convert the proposals to keep back to a numpy array
        proposals = np.array(proposals_to_keep)
    return suppressed_detections

def camera_proposal_decision(camera_proposal):
    # some objects are detected by multiple cameras, we need to decide which one to use
    proposals = []
    for key, centers in camera_proposal.items():
        # filter out the empty centers
        valid_centers = [center for center in centers if len(center) > 0]
        for cnt in valid_centers:
            proposals.append(cnt)
    if proposals:
        proposals = np.array(proposals)
    else:
        proposals = np.array([]).reshape(-1, 6)
    print('raw proposals: \n', proposals)
     # Initialize a list to keep the suppressed detections
    
    suppressed_detections = []
    # get the unique labels
    if len(proposals):
        unique_labels = np.unique(proposals[:, 3])
        for label in unique_labels:
            class_proposals = proposals[proposals[:, 3] == label]
            suppresed_class_proposals = per_class_suppression(class_proposals)
            if len(suppresed_class_proposals) > 0:
                suppressed_detections.extend(suppresed_class_proposals)
        # convert label 0, 1, 3 to '1' -- indicating the person
    suppressed_detections = np.array(suppressed_detections)
    return np.array(suppressed_detections).reshape(-1, 6) # 2d array

#================================================ 
# refine the conventional pipeline result
ISC_NAME_TO_CLASS = {
    'VRU_Adult_Using_Motorized_Bicycle': 0,
    'Passenger_Vehicle': 1,
    'VRU_Child': 2,
    'VRU_Adult': 3,
    'VRU_Adult_Using_Cane': 4,
    'VRU_Adult_Using_Manual_Scooter': 5,
    'VRU_Adult_Using_Crutches': 6,
    'VRU_Adult_Using_Cardboard_Box': 7,
    'VRU_Adult_Using_Walker': 8,
    'VRU_Adult_Using_Manual_Wheelchair': 9,
    'VRU_Adult_Using_Stroller': 10,
    'VRU_Adult_Using_Skateboard': 11,
    'VRU_Adult_Using_Manual_Bicycle': 12,
    }

def load_conventional_pipeline_result(Run_num):
      #labels
    file_format = f'../Intersection-Safety-Challenge/conventional_pipeline/sample_detections_validation/Run_{Run_num}_detections_oriented_lidar2.txt'
    #convert the txt to a pandas dataframe
    df = pd.read_csv(file_format, header=None, names=['bin_index', 'label', 'x', 'x_len','y', 'y_len','z', 'z_len', 'rotation','score'])
    df['sup_label'] = df['label'].apply(lambda x: x.split('_')[0]) # 'vru', 'vehicle'
    df['sup_label'] = df['sup_label'].apply(lambda x: 'Vehicle' if x == ' Passenger' else x) # 这里一个空格
    return df

def get_one_bin_detection(df, bin_index):
    # get the detection for one bin
    # df: the dataframe containing all the detections
    # bin_index: the index of the bin
    # return: the dataframe containing the detection for one bin
    return df[df['bin_index'] == bin_index]

# 
# load_conventional_pipeline_result(Run_num=55)
def refine_conventional_pipeline_result(df,  bin_index, voted_camera_proposal):
    refined_results = [] # 'x', 'y', 'z', 'label','score','roration'
    conventional_result = get_one_bin_detection(df, f'{bin_index:06d}.pcd')

    mask = voted_camera_proposal[:, 3] == 0 # person
    voted_cam_vehicles = voted_camera_proposal[~mask]
    voted_cam_vrus = voted_camera_proposal[mask]

    convention_vehicles = conventional_result[conventional_result['sup_label'] == 'Vehicle']
    conventaion_vehicles_centers = convention_vehicles[['x', 'y', 'z', 'label','score','rotation']].values

    convention_vrus = conventional_result[conventional_result['sup_label'] == ' VRU'] # VRU 前面空格
    conventaion_vrus_centers = convention_vrus[['x', 'y', 'z', 'label','score','rotation']].values
    
    # set the voted_vehicles to the refined vehicles, because camera is more accurate than conventional LiDAR
    # but if the background filter is not good, the camera may detect some false positive
    if len(voted_cam_vehicles) > 0:
        refined_results.append(voted_cam_vehicles)
    else:
        refined_results.append(conventaion_vehicles_centers)
    
    # if take the vru detection from the conventional pipeline
    if len(convention_vrus) > 0:
        if len(voted_cam_vrus) > 0:
            # add the score of the voted camera proposal to the conventional pipeline by calulating the average score
            conventaion_vrus_centers[:, 4] =  voted_cam_vrus[:, 4].mean()
        refined_results.append(conventaion_vrus_centers)
    else:
        # if the conventional pipeline has no VRU detection, take the VRU detection from the camera
        if len(voted_cam_vrus) > 0:
            refined_results.append(voted_cam_vrus)
    
    refined_results = [item.tolist() for sublist in refined_results for item in sublist]
    for item in refined_results:
        if item[3] in CLUSTER_LABELS:
            item[3] = CLUSTER_LABELS[item[3]]
    # print('refused results',refined_results)
    return refined_results

# refine_conventional_pipeline_result(load_conventional_pipeline_result(Run_num=55), 24, 0)



#=====================================================
# test the synchronization between the camera and LiDAR data
def test_synchronization(Run_num, sync_one_frame = False, selected_frame = None, sync_one_cam = False, selected_cam = None):
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
    fusion_dir = f'{run_folder_dir}/fusion' # save the fusion result to a folder
    os.makedirs(fusion_dir, exist_ok=True)

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

    # The scenario is: with all synced cameras and the lidar frame, we want to get the detection from the camera and the lidar points
    
    if sync_one_frame:
        syn_bins = [selected_frame]
        cameras = np.arange(1, 6)

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
        out = cv2.VideoWriter(f'{fusion_dir}/Camera{4}_all.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        syn_bins = np.arange(first_syn, last_syn+1).astype(int)
        cameras = [selected_cam]
    

    for bin_idx in syn_bins:
        print(f'Processing lidar frame {bin_idx}')
        # get the frame from the lidar2
        lidar2_frame_path = lidar_path_format.format(bin_idx)
        # work on camera 4 first
        # create eight subplots to show the images each ax with size 10,5 
        if sync_one_frame:
            fig, axs = plt.subplots(3,2 , figsize=(20, 10))
        
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
            
            
            bboxes, labels, scores, centers, velo_uvz = get_fused_detection(image= frame, 
                                                                        bin_path= lidar2_frame_path, 
                                                                        model= inferencer, K= K, D= D, 
                                                                        T_velo_cam= T_velo_cam, T_cam_velo = T_cam_velo, draw_depth= False)
            # print('velo_uvz', velo_uvz)
            
            velo_image = draw_velo_on_image(velo_uvz, frame)

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
            plt.savefig(f'{fusion_dir}/Lidar_frame{bin_idx}_summary.jpg')


        print(f'Camera {i} frame {bin_idx} is processed successfully')
