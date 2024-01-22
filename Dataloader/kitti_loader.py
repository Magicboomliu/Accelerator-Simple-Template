from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset

import os
import sys
from Dataloader.kitti_io import read_img
from Dataloader.calib_parse import parse_calib
from utils.file_io import read_text_lines
import numpy as np

def cut_or_pad_img(img, targetHW):

    t_H, t_W = targetHW
    H, W = img.shape[0], img.shape[1]

    padW = np.abs(t_W - W)
    half_padW = int(padW//2)
    # crop
    if W > t_W:
        img = img[:, half_padW:half_padW+t_W]
    # pad
    elif W < t_W:
        img = np.pad(img, [(0, 0), (half_padW, padW-half_padW), (0, 0)], 'constant')

    # crop
    padH = np.abs(t_H - H)
    if H > t_H:
        img = img[padH:, :]
    # pad
    elif H < t_H:
        padH = t_H - H
        img = np.pad(img, [(padH, 0), (0, 0), (0, 0)], 'constant')

    return img







class KITTIRaw_Dataset(Dataset):
    
    def __init__(self,datapath,
                 trainlist,vallist,
                 mode='train',
                 transform=None,
                 targetHW = (370,1232),
                 save_filename=False):
        super(KITTIRaw_Dataset,self).__init__()
        
        self.datapath = datapath
        self.trainlist = trainlist
        self.vallist = vallist
        self.mode = mode
        self.transform = transform
        self.save_filename = save_filename
        self.train_resolution = targetHW
        
        
        dataset_dict = {
            'train': self.trainlist,
            'val': self.vallist,
            "test": self.vallist
        }
        
        self.samples  =[]
        
        lines = read_text_lines(dataset_dict[mode])
        
        for line in lines:
            splits = line.split()
            left_image_path = splits[0]
            right_image_path = left_image_path.replace("image_02",'image_03')
            # camera_pose = os.path.join(left_image_path[:10],"calib_cam_to_cam.txt") 
            sample = dict()
            
            if self.save_filename:
                sample['left_name']= left_image_path.replace("/","_")
            
            sample['left_image_path'] = os.path.join(datapath,left_image_path)
            sample['right_image_path'] = os.path.join(datapath,right_image_path)
            
            
            # sample['camera_pose_path'] = os.path.join(datapath,camera_pose)
            
            self.samples.append(sample)
    
    
    def __getitem__(self, index):

        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']
        
        
        sample['left_image'] = read_img(sample_path['left_image_path'])
        sample['right_image'] = read_img(sample_path['right_image_path'])
        
        
        if self.train_resolution is not None:
            sample['left_image'] = cut_or_pad_img(sample['left_image'],targetHW=self.train_resolution)
            sample['right_image'] = cut_or_pad_img(sample['right_image'],targetHW=self.train_resolution)
        
        
        sample['left_camera_pose'] = np.array([[1,0,0,0],
                                              [0,1,0,0],
                                              [0,0,1,0],
                                              [0,0,0,1]]).astype(np.float32)
        sample['right_camera_pose'] = np.array([[1,0,0,-0.537165],
                                                [0,1,0,0],
                                                [0,0,1,0],
                                                [0,0,0,1]]).astype(np.float32)
        
    
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)
        