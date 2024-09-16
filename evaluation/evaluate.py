# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import io as sysio
import sys
import os
from isc_common import get_label_annos

# sys.path.insert(0, os.path.abspath('/home/haishan/Projects/Intersection_safety/Intersection_Safety_Challenge/Intersection-Safety-Challenge/deep_learing_pipeline/'))
# from mmdet3d.evaluation.functional.kitti_utils.eval import do_eval, do_coco_style_eval
from mmdet3d.evaluation.functional.kitti_utils.isc_eval import do_eval, do_coco_style_eval

def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

def get_gt_anno():
    gt_name = np.array(
        ['VRU_Adult', 'VRU_Adult_Using_Manual_Bicycle'])
    gt_truncated = np.zeros((len(gt_name)), dtype=float)
    gt_occluded = np.zeros((len(gt_name)), dtype=int)
    gt_alpha = np.zeros((len(gt_name)), dtype=float)
    gt_bbox = np.zeros((len(gt_name), 4), dtype=float)
    gt_dimensions = np.array([[1.20316797265858, 1.12126152822847, 1.97323795370992], 
                              [1.11762419762236, 2.06065880175287, 1.58873158152365]])
    gt_location = np.array([[6.89494617578582, 3.845501155986, 3.21665861103567],
                            [8.93528057436735, 3.58458457558635, 3.52193936976064]])
    gt_rotation_y = np.array([255.411113507261/180*np.pi, 296.041815657672/180*np.pi])
    gt_anno = dict(
        name=gt_name,
        truncated=gt_truncated,
        occluded=gt_occluded,
        alpha=gt_alpha,
        bbox=gt_bbox,
        dimensions=gt_dimensions,
        location=gt_location,
        rotation_y=gt_rotation_y)
    return gt_anno

# just an example, not the actual detection results
def get_dt_anno():
    dt_name = np.array(['VRU_Adult', 'VRU_Adult_Using_Manual_Bicycle'])
    dt_truncated = np.zeros((len(dt_name)), dtype=float)
    dt_occluded = np.zeros((len(dt_name)), dtype=int)
    dt_alpha = np.zeros((len(dt_name)), dtype=float)
    dt_dimensions = np.array([[1.2316797265858, 1.1126152822847, 1.9323795370992],
                              [1.1762419762236, 2.0065880175287, 1.5873158152365]])
    dt_location = np.array([[6.8494617578582, 3.85501155986, 3.2665861103567],
                            [8.9528057436735, 3.5458457558635, 3.5193936976064]])
    dt_rotation_y = np.array([255.1113507261/180*np.pi, 296.1815657672/180*np.pi])
    dt_bbox = np.zeros((len(dt_name), 4), dtype=float)
    dt_score = np.array(
        [0.8, 0.9])
    dt_anno = dict(
        name=dt_name,
        truncated=dt_truncated,
        occluded=dt_occluded,
        alpha=dt_alpha,
        bbox=dt_bbox,
        dimensions=dt_dimensions,
        location=dt_location,
        rotation_y=dt_rotation_y,
        score=dt_score)
    return dt_anno

def test_do_eval(gt_annos, dt_annos):
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and CUDA')
   
    current_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    overlap_0_7 = np.array([[0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    overlap_0_5 = np.array([[0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0) 

    class_to_name = {
        0: 'VRU_Adult_Using_Motorized_Bicycle',
        1: 'Passenger_Vehicle',
        2: 'VRU_Child',
        3: 'VRU_Adult',
        4: 'VRU_Adult_Using_Cane',
        5: 'VRU_Adult_Using_Manual_Scooter',
        6: 'VRU_Adult_Using_Crutches',
        7: 'VRU_Adult_Using_Cardboard_Box',
        8: 'VRU_Adult_Using_Walker',
        9: 'VRU_Adult_Using_Manual_Wheelchair',
        10: 'VRU_Adult_Using_Stroller',
        11: 'VRU_Adult_Using_Skateboard',
        12: 'VRU_Adult_Using_Manual_Bicycle',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]

    result = ''
    
    eval_types = ['bev', '3d']

    mAP11_bbox, mAP11_bev, mAP11_3d, mAP11_aos, mAP40_bbox,\
        mAP40_bev, mAP40_3d, mAP40_aos = do_eval(gt_annos, dt_annos,
                                                 current_classes, min_overlaps,
                                                 eval_types)

    difficulty = ['easy', 'moderate', 'hard']
    # Calculate AP40
    result += '\n----------- AP40 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP40@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP40_bev is not None:
                result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_bev[j, :, i])
            if mAP40_3d is not None:
                result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_3d[j, :, i])

    # calculate mAP40 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP40@{}, {}, {}:\n'.format(*difficulty))
        if mAP40_bev is not None:
            mAP40_bev = mAP40_bev.mean(axis=0)
            result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP40_bev[:, 0])
        if mAP40_3d is not None:
            mAP40_3d = mAP40_3d.mean(axis=0)
            result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP40_3d[:,
                                                                            0])
    print(f'AP Results:\n' + result)
    return result

def test_do_coco_style_eval(gt_annos, dt_annos):

    current_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
    class_to_name = {
        0: 'VRU_Adult_Using_Motorized_Bicycle',
        1: 'Passenger_Vehicle',
        2: 'VRU_Child',
        3: 'VRU_Adult',
        4: 'VRU_Adult_Using_Cane',
        5: 'VRU_Adult_Using_Manual_Scooter',
        6: 'VRU_Adult_Using_Crutches',
        7: 'VRU_Adult_Using_Cardboard_Box',
        8: 'VRU_Adult_Using_Walker',
        9: 'VRU_Adult_Using_Manual_Wheelchair',
        10: 'VRU_Adult_Using_Stroller',
        11: 'VRU_Adult_Using_Skateboard',
        12: 'VRU_Adult_Using_Manual_Bicycle',
    }

    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.5, 0.95, 10],
        2: [0.5, 0.95, 10],
        3: [0.5, 0.95, 10],
        4: [0.5, 0.95, 10],
        5: [0.5, 0.95, 10],
        6: [0.5, 0.95, 10],
        7: [0.5, 0.95, 10],
        8: [0.5, 0.95, 10],
        9: [0.5, 0.95, 10],
        10: [0.5, 0.95, 10],
        11: [0.5, 0.95, 10],
        12: [0.5, 0.95, 10],
    }

    name_to_class = {v: n for n, v in class_to_name.items()}

    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(class_to_range[curcls])[:,
                                                                   np.newaxis]
    
    result = ''


    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print((f'{class_to_name[curcls]} '
                             'coco AP@{:.2f}:{:.2f}:{:.2f}:'.format(*o_range)))
        result += print_str((f'bev  AP:{mAPbev[j, 0]:.2f}, '
                             f'{mAPbev[j, 1]:.2f}, '
                             f'{mAPbev[j, 2]:.2f}'))
        result += print_str((f'3d   AP:{mAP3d[j, 0]:.2f}, '
                             f'{mAP3d[j, 1]:.2f}, '
                             f'{mAP3d[j, 2]:.2f}'))
    
    return result

def main():
    # gt path and detection path
    gt_loc = '../datasets/validation_data_full/kitti_GT_2/Run_48/Run_48_GT.txt'
    dt_loc = '../camera_fused_label/fused_label_lidar12_cam24/masked_fusion_label_coco/Run_48_fused_result.txt'
    # TBD: extract gt_annos and dt_annos, then run the evaluation pipeline
    # print(dt_anno)
    # rst = test_do_eval([gt_anno], [dt_anno])
    # rst = test_do_eval(gt_annos, dt_annos)
    # rst = test_do_coco_style_eval(gt_annos, dt_annos)

    dataset_dir = '/home/haishan/Projects/Intersection_safety/Intersection_Safety_Challenge/datasets/validation_data_full'
    run_nums = np.sort(np.array([int(dir.split('_')[-1]) for dir in os.listdir(dataset_dir) if dir.startswith('Run_')]))
    # print(run_nums)
    full_gt_annos = []
    full_dt_annos = []
    for run_num in run_nums:
        gt_annos, dt_annos = get_label_annos(run_num)
        full_gt_annos.extend(gt_annos)
        full_dt_annos.extend(dt_annos)
    assert len(gt_annos) == len(dt_annos)
    print(len(full_gt_annos))
    print(len(full_dt_annos))
    rst = test_do_eval(full_gt_annos, full_gt_annos)

if __name__ == "__main__":
    main()
