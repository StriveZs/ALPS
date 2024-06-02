# --------------------------------------------------------
# ALPS
# Copyright (c)
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import joblib
import numpy as np
import cv2
import sys
import os
import glob
import json

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms as T

# Dataset For SAM
class Custom_Dataset(Dataset):
    def __init__(self, root_dir, mask_dir, dataset_path, prefix, image_suffix):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.prefix = prefix
        self.image_suffix = image_suffix
        self.generate_meta_list()
        self.define_transforms()
        self.scale_factor = 32 # 896 / 32
        self.dataset_path = dataset_path
    
    def generate_meta_list(self,):
        img_list = glob.glob(os.path.join(self.root_dir, '*'+self.image_suffix))
        self.meta_list = img_list
    
    def __len__(self,):
        return len(self.meta_list)
    
    def get_all_masks(self, img_name):
        mask_path = os.path.join(self.mask_dir, img_name+'_'+self.prefix)
        mask_list = glob.glob(os.path.join(mask_path, '*'+self.image_suffix))
        masks_list = []
        for mask_path in mask_list:
            mask_name = mask_path.split('/')[-1].split('.')[0] # img name
            if 'full_mask' in mask_name:
                continue
            mask = np.expand_dims(np.array(cv2.imread(mask_path, 0) / 255.0, dtype=np.float32), axis=-1) # 255->1 expand_dims:[H, W] -> [H, W, 1]
            mask = self.transform(mask)
            pair = dict()
            pair['mask'] = mask
            pair['mask_name'] = mask_name # record id
            masks_list.append(pair)
        return masks_list
    
    def define_transforms(self):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(1024), 
            T.ToTensor(),
        ])
    
    def __getitem__(self, index):
        item = self.meta_list[index]
        img_path = os.path.join(self.dataset_path, item)
        image = cv2.imread(img_path)
        image = self.transform(image)
        # image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.float32)
        img_name = item.split('/')[-1].split('.')[0] # img name
        masks = self.get_all_masks(img_name)
        return {'image_name': img_name,
                'image': image,
                'masks': masks,
                }


# Mask Vec Dataset For Online K-means Train Dataset
class Mask_Vec_Dataset(Dataset):
    def __init__(self, root_dir, json_dir, item_dir):
        self.root_dir = root_dir
        self.json_dir = json_dir
        self.item_dir = item_dir
        self.generate_meta_list()
    
    def generate_meta_list(self,):
        self.meta_list = []
        json_list = glob.glob(os.path.join(self.root_dir, self.json_dir, self.item_dir ,'*.json'))
        for json_path in json_list:
            with open(json_path,'r', encoding='UTF-8') as f:
                data = json.load(f)
            image_name = data['image_name']
            mask_vec_dict = data['mask_avg_vec']
            for key in mask_vec_dict.keys():
                mask_name = key
                mask_vec_path = mask_vec_dict[key]
                self.meta_list.append([image_name, mask_name, mask_vec_path])
    
    def __len__(self,):
        return len(self.meta_list)
    
    def get_mask_vec(self, filename):
        return np.load(filename)
    
    def __getitem__(self, index):
        item = self.meta_list[index]
        _, _, mask_vec_path = item[0], item[1], item[2]
        mask_vec = self.get_mask_vec(mask_vec_path)
        return np.array(mask_vec)


# Mask Vec Dataset For Online K-means Test Dataset
class Mask_Vec_TestDataset(Dataset):
    def __init__(self, root_dir, json_dir, item_dir):
        self.root_dir = root_dir
        self.json_dir = json_dir
        self.item_dir = item_dir
        self.generate_meta_list()
    
    def generate_meta_list(self,):
        self.meta_list = []
        json_list = glob.glob(os.path.join(self.root_dir, self.json_dir, self.item_dir ,'*.json'))
        for json_path in json_list:
            with open(json_path,'r', encoding='UTF-8') as f:
                data = json.load(f)
            image_name = data['image_name']
            mask_vec_dict = data['mask_avg_vec']
            for key in mask_vec_dict.keys():
                mask_name = key
                mask_vec_path = mask_vec_dict[key]
                self.meta_list.append([image_name, mask_name, mask_vec_path])
    
    def __len__(self,):
        return len(self.meta_list)
    
    def get_mask_vec(self, filename):
        return np.load(filename)
    
    def __getitem__(self, index):
        item = self.meta_list[index]
        image_name, mask_name, mask_vec_path = item[0], item[1], item[2]
        mask_vec = self.get_mask_vec(mask_vec_path)
        return {
            "image_name": image_name,
            "mask_name": mask_name,
            "mask_vec": np.array(mask_vec)
        }