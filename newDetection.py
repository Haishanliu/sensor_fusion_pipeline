import cv2
import os
import numpy as np
import pandas as pd
from mmdet.apis import DetInferencer
import time

from concurrent.futures import ThreadPoolExecutor

def compute_inclusion_ratio(box1, box2):
    """
    Compute the proportion of box1's area that is covered by box2.

    Parameters
    ----------
    box1 : [x1, y1, x2, y2] New bounding box
    box2 : [x1, y1, x2, y2] Existing bounding box

    Returns
    -------
    float
        Inclusion ratio, ranging between [0, 1].
    """
    x1, y1, x2, y2 = box1  # New bounding box
    x1g, y1g, x2g, y2g = box2  # Existing bounding box

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = max(0, x2 - x1) * max(0, y2 - y1)
    if box1_area == 0:
        return 0.0

    inclusion_ratio = inter_area / box1_area
    return inclusion_ratio

def merge_overlapping_boxes(detections):
    """
    Merge overlapping bounding boxes into a single bounding box.

    Parameters
    ----------
    detections : list of dict
        List of detection results, each element is a dictionary containing 'label', 'score', 'bbox', 'method'.

    Returns
    -------
    merged_detections : list of dict
        List of merged detection results.
    """
    merged_detections = []
    while detections:
        base_det = detections.pop(0)
        base_bbox = base_det['bbox']
        overlapping_dets = [base_det]
        non_overlapping_dets = []
        for det in detections:
            bbox = det['bbox']
            # Check if there is overlap (IoU > 0)
            iou = compute_iou(base_bbox, bbox)
            if iou > 0:
                overlapping_dets.append(det)
            else:
                non_overlapping_dets.append(det)
        # Merge overlapping bounding boxes
        x1s = [det['bbox'][0] for det in overlapping_dets]
        y1s = [det['bbox'][1] for det in overlapping_dets]
        x2s = [det['bbox'][2] for det in overlapping_dets]
        y2s = [det['bbox'][3] for det in overlapping_dets]
        merged_bbox = [min(x1s), min(y1s), max(x2s), max(y2s)]
        # Select the detection result with the highest confidence
        best_det = max(overlapping_dets, key=lambda d: d['score'])
        merged_detections.append({
            'label': best_det['label'],
            'score': best_det['score'],
            'bbox': merged_bbox,
            'method': best_det['method']
        })
        detections = non_overlapping_dets
    return merged_detections

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : [x1, y1, x2, y2]
    box2 : [x1, y1, x2, y2]

    Returns
    -------
    float
        IoU value, ranging between [0, 1].
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = max(0, x2 - x1) * max(0, y2 - y1)
    box2_area = max(0, x2g - x1g) * max(0, y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

# Define frame folder and result path
# frames_folder = 'CAM4_214/'
# results_path = 'Results_214/'

def multi_methods_detection(Run_num, cam_num, debug=False):
    dataset_dir = '../datasets/validation_data_full'
    run_folder_dir = f'{dataset_dir}/Run_{Run_num}'
    # only process the frames that are synced with the lidar2
    cam_frame_timing_format = f'VisualCamera{{}}_Run_{{}}_frame-timing_sync.csv'
    video_format = f'VisualCamera{{}}_Run_{Run_num}.mp4'

    video_path = os.path.join(run_folder_dir, video_format.format(cam_num))
    cap = cv2.VideoCapture(video_path)

    results_path = os.path.join(run_folder_dir, f'VisualCamera{cam_num}_2d_Results_sync')
    os.makedirs(results_path, exist_ok=True)

    if not cap.isOpened():
        print(f"Unable to open video {video_path}")
        raise ValueError(f"Unable to open video {video_path}")


    '''image is the read image from the camera'''
    # Read method configurations
    with open('methods.txt', 'r') as file:
        method_configs = file.read().splitlines()

    # Assign a unique color to each method
    method_colors = {}
    color_palette = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Yellow
        (128, 0, 128), # Purple
        (128, 128, 0), # Olive
        (0, 128, 128), # Teal
        (128, 0, 0),   # Maroon
        # Add more colors if there are more methods
    ]
  
    for idx, method_config in enumerate(method_configs):
        method_name = method_config.split('_')[0]
        method_colors[method_name] = color_palette[idx % len(color_palette)]

    # Initialize the potential bounding boxes from the previous frame
    previous_potential_bboxes = []
    cam_frame_timing = pd.read_csv(os.path.join(run_folder_dir, cam_frame_timing_format.format(cam_num, Run_num)))
    # drop the nan values in the Lidar2_frame_idx
    cam_frame_timing = cam_frame_timing.dropna(subset=['Lidar2_frame_idx'])
    for bin_idx in cam_frame_timing['Lidar2_frame_idx'].unique():
        
        sync_image_num = cam_frame_timing.loc[cam_frame_timing['Lidar2_frame_idx'] == bin_idx, 'Image_number'].values[-1]
       
        cap.set(cv2.CAP_PROP_POS_FRAMES,sync_image_num)
        ret, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # image_path = os.path.join(frames_folder, image_file)
        # image = cv2.imread(image_path)
        # height, width = image.shape[:2]
        image = frame

        # Initialize the combined detection result list
        combined_detections = []

        # Copy of the image for drawing bounding boxes
        drawn_image = image.copy()

        for method_config in method_configs:
            # Initialize DetInferencer
            inferencer = DetInferencer(method_config)

            # Get method name and corresponding color
            method_name = method_config.split('_')[0]
            method_color = method_colors[method_name]

            # Perform detection
            results = inferencer(image, pred_score_thr=0.1, show=False, no_save_pred=True, return_vis=False)

            # Process results
            for result in results['predictions']:
                labels = result['labels']
                scores = result['scores']
                bboxes = result['bboxes']  # [x1, y1, x2, y2]
                for j in range(len(labels)):
                    label = labels[j]
                    score = scores[j]
                    bbox = bboxes[j]
                    if score >= 0.3 and label != 9 and label != 4:
                        # Check if the new box is covered at least 70% by existing boxes
                        overlap = False
                        for existing_det in combined_detections:
                            existing_bbox = existing_det['bbox']
                            inclusion_ratio = compute_inclusion_ratio(bbox, existing_bbox)
                            if inclusion_ratio >= 0.7:
                                overlap = True
                                break
                        if not overlap:
                            # Add new detection result
                            combined_detections.append({
                                'label': label,
                                'score': score,
                                'bbox': bbox,
                                'method': method_name
                            })

        # Merge overlapping bounding boxes using multiple methods
        merged_detections = merge_overlapping_boxes(combined_detections)

        # Initialize the potential bounding boxes for the current frame
        current_potential_bboxes = []
        valid_detections = []

        if idx < 3:
            # For the first three frames, all detection results are directly considered valid bounding boxes
            for det in merged_detections:
                det['level'] = 2  # Directly set to level 2
                current_potential_bboxes.append(det)
                valid_detections.append(det)
        else:
            # For the 4th frame and beyond, perform potential bounding box processing
            for det in merged_detections:
                bbox = det['bbox']
                label = det['label']
                score = det['score']
                method_name = det['method']
                # Compare with the potential bounding boxes from the previous frame
                max_level = -1
                for prev_det in previous_potential_bboxes:
                    prev_bbox = prev_det['bbox']
                    prev_level = prev_det['level']
                    iou = compute_iou(bbox, prev_bbox)
                    if iou > 0:
                        if prev_level > max_level:
                            max_level = prev_level
                # Set the level of the potential bounding box
                level = max_level + 1
                # Create potential bounding box
                potential_bbox = {
                    'label': label,
                    'score': score,
                    'bbox': bbox,
                    'method': method_name,
                    'level': level
                }
                current_potential_bboxes.append(potential_bbox)
                if level >= 2:
                    valid_detections.append(potential_bbox)

        # Update the potential bounding boxes from the previous frame
        previous_potential_bboxes = current_potential_bboxes

        # Clear the drawn image and redraw valid bounding boxes
        drawn_image = image.copy()
        for det in valid_detections:
            bbox = det['bbox']
            label = det['label']
            score = det['score']
            method_name = det['method']
            method_color = method_colors[method_name]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), method_color, 2)
            cv2.putText(drawn_image, f"{label}:{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_color, 2)

        if debug:
            # Save the image with detection results
            result_image_path = os.path.join(results_path, str(sync_image_num) + '.jpg')
            cv2.imwrite(result_image_path, drawn_image)

        # Save the valid detection results to a CSV file
        csv_file_path = os.path.join(results_path, str(sync_image_num) + '.csv')
        df = pd.DataFrame(valid_detections)
        if not df.empty:
            # Convert bbox to x, y, w, h
            df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
            # df['x'] = df['x1']
            # df['y'] = df['y1']
            # df['w'] = df['x2'] - df['x1']
            # df['h'] = df['y2'] - df['y1']
            df = df[['label', 'score', 'x1', 'y1', 'x2', 'y2', 'method']]
            df.to_csv(csv_file_path, index=False)
           
        else:
            # If there are no detection results, save an empty CSV file
            df = pd.DataFrame(columns=['label', 'score', 'x', 'y', 'w', 'h', 'method'])
            df.to_csv(csv_file_path, index=False)

def run_detection(Run_num, cam_num, debug=False):
    t_0 = time.time()
    multi_methods_detection(Run_num, cam_num, debug)
    t_1 = time.time()
    print(f"Run_num: {Run_num}, Cam_num: {cam_num}, Time taken: {t_1- t_0:.2f} seconds")


if __name__ == '__main__':
    run_detection(850, 4, debug=True)
    # run_nums= [int(dir.split('_')[1]) for dir in os.listdir('../datasets/validation_data_full') if dir.startswith('Run_')]
    # print(run_nums)
    # # print(a)
    # cam_nums = [2, 4]
    # t_0 = time.time()
    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     for Run_num in run_nums:
    #         for cam_num in cam_nums:
    #             executor.submit(run_detection, Run_num, cam_num)
  
