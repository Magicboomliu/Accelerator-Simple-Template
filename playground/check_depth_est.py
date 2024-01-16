import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import sys
sys.path.append("..")

from dataloader.sceneflow_loader import StereoDataset
from torch.utils.data import DataLoader
from dataloader import transforms
import os

from utils.common import logger


# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Get Dataset Here
def prepare_dataset(data_name,
                    datapath=None,
                    trainlist=None,
                    vallist=None,
                    batch_size=1,
                    test_batch=1,
                    datathread=4,
                    logger=None):
    
    # set the config parameters
    dataset_config_dict = dict()
    
    if data_name == 'sceneflow':
        train_transform_list = [
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        train_transform = transforms.Compose(train_transform_list)

        val_transform_list = [transforms.ToTensor(),
                        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        val_transform = transforms.Compose(val_transform_list)
        
        train_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                dataset_name='SceneFlow',mode='train',transform=train_transform)
        test_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                dataset_name='SceneFlow',mode='val',transform=val_transform)

    img_height, img_width = train_dataset.get_img_size()


    datathread=4
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    
    if logger is not None:
        logger.info("Use %d processes to load data..." % datathread)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    
    num_batches_per_epoch = len(train_loader)
    
    
    dataset_config_dict['num_batches_per_epoch'] = num_batches_per_epoch
    dataset_config_dict['img_size'] = (img_height,img_width)
    
    
    return (train_loader,test_loader),dataset_config_dict


if __name__=="__main__":
    

    datapath = "/data1/liu/"
    trainlist = "/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/SceneFlow_With_Occ.list"
    vallist = "/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list"
    
    
    (train_loader,test_loader), dataset_config_dict = prepare_dataset(data_name='sceneflow',
                                                                      datapath=datapath,trainlist=trainlist,
                                                                      vallist=vallist,batch_size=1,
                                                                      test_batch=1,datathread=4,logger=logger)
    
    for idx, sample in enumerate(train_loader):
        left_img = sample['img_left']
        right_img = sample['img_right']
        left_disp = sample['gt_disp']    
        pass
    
    
    
    
    
    
    