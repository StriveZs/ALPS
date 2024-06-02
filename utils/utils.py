# --------------------------------------------------------
# ALPS
# Copyright (c)
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import glob
import json
import gc
import joblib
import time
import datetime
import os
import os.path as osp

#----------------Step 1----------------
## visual
def save_anns(image, anns, save_path):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

## filter masks by shreshold
def filter_masks(masks, threshold):
    filter_masks = []
    total = masks[0]['segmentation'].shape[0] * masks[0]['segmentation'].shape[1]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        # filter
        total_value = np.sum(mask)
        percentage = total_value / total
        if percentage > threshold:
            continue
        filter_masks.append(mask_data)
    return filter_masks

## save each instance mask for one img
def write_masks_to_folder(image, masks, path, image_suffix, threshold=None):
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    # filter masks
    if threshold is not None:
        masks = filter_masks(masks, threshold)
    # save visual annotaion fig
    save_anns(image, masks, os.path.join(path, 'full_mask'+image_suffix))
    # save each mask
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}"+image_suffix
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return

# generate color map for visualization
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

# create instance color mapping
def create_color_mapping(maximum=255):
    assert maximum in [255, 1], maximum
    color_id = _COLORS * maximum
    len_color = _COLORS.shape[0] # 74
    # print(len_color)
    instance_id = [i for i in range(len_color)]
    color_map = dict(zip(instance_id, color_id))
    return color_map

# generate pseudo mask
def add_mask_to_template(mask, template, color_value):
    mask = np.expand_dims(mask, axis=-1)
    mask = mask.repeat(3, axis=-1)
    # mask = mask * color_value
    for i in range(len(mask.shape)):
        mask[:,:,i] = mask[:,:,i] * color_value[i]
    # template += mask
    nozero_index = np.transpose(np.nonzero(mask))
    # print(nozero_index)
    # 采用覆盖，而不是直接累加
    for index in nozero_index:
        template[index[0], index[1], index[2]] = mask[index[0], index[1], index[2]]
    return template
    

def add_gt_seg_mask_to_template(mask, template, mask_type):
    mask[:,:] = mask[:,:] * mask_type
    mask = np.expand_dims(mask, axis=-1)
    # template += mask
    # print(mask.shape)
    nozero_index = np.transpose(np.nonzero(mask))
    # print(nozero_index)
    for index in nozero_index:
        template[index[0], index[1], index[2]] = mask[index[0], index[1], index[2]]
    return template