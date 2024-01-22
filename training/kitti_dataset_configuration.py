import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from Dataloader.kitti_loader_simple import KITTIRaw_Dataset
from Dataloader import transforms 
import os
import logging


# Get Dataset Here
def prepare_dataset(datapath,
                    trainlist,
                    vallist,
                    logger=None,
                    batch_size = 1,
                    test_size =1,
                    datathread = 4
                    ):
    
    train_transform_list = [transforms.ToTensor(),]
    train_transform = transforms.Compose(train_transform_list)

    val_transform_list = [transforms.ToTensor()]
    
    val_transform = transforms.Compose(val_transform_list)
    
    
    train_dataset = KITTIRaw_Dataset(datapath=datapath,trainlist=trainlist,vallist=vallist,transform=train_transform,
                                     mode='train')

    test_dataset = KITTIRaw_Dataset(datapath=datapath,trainlist=trainlist,vallist=vallist,transform=val_transform,
                                     mode='test')


    datathread=datathread
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_size, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)    
    return (train_loader,test_loader),num_batches_per_epoch




def resize_max_res_tensor(input_tensor,is_disp=False,recom_resolution=768):
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         scale_factor=downscale_factor,mode='bilinear',
                                         align_corners=False)
    
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor
    

def image_normalization(image_tensor):
    image_normalized = image_tensor * 2.0 -1.0
    # means = torch.ones([3]).unsqueeze(0).unsqueeze(0).permute(2,0,1) * 0.5
    # vals = torch.ones([3]).unsqueeze(0).unsqueeze(0).permute(2,0,1) * 0.5
    
    # means = means.type_as(image_tensor)
    # vals = vals.type_as(image_tensor)
    
    # image_normalized = (image_tensor - means)/vals
    return image_normalized