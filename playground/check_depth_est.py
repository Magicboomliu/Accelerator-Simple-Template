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
from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps


# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

from diffusers import AutoencoderKL

from utils.de_normalized import de_normalization


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

def Disparity_Normalization(disparity):
    min_value = torch.min(disparity)
    max_value = torch.max(disparity)
    normalized_disparity = ((disparity -min_value)/(max_value-min_value+1e-5) - 0.5) * 2    
    return normalized_disparity

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
    




    
    




if __name__=="__main__":
    

    datapath = "/data1/liu/"
    trainlist = "/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/SceneFlow_With_Occ.list"
    vallist = "/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list"
    
    
    (train_loader,test_loader), dataset_config_dict = prepare_dataset(data_name='sceneflow',
                                                                      datapath=datapath,trainlist=trainlist,
                                                                      vallist=vallist,batch_size=1,
                                                                      test_batch=1,datathread=4,logger=logger)
    
    pretrained_model_name_path = "stabilityai/stable-diffusion-2"
        
    # define the vae
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.cuda()
    print("Loaded the VAE pre-trained model successfully!")
    
    
    for idx, sample in enumerate(train_loader):
        left_img = sample['img_left']
        right_img = sample['img_right']
        left_disp_single = sample['gt_disp']
        
        left_disp_single = left_disp_single.unsqueeze(0)
        left_disp = left_disp_single.repeat(1,3,1,1)
        
        resized_left_disp = resize_max_res_tensor(left_disp,is_disp=True)
        
        normaliazed_left_disp = Disparity_Normalization(resized_left_disp)
        
        normaliazed_left_disp = normaliazed_left_disp.cuda()
        
        with torch.no_grad():
            latents = vae.encode(normaliazed_left_disp).latent_dist.sample()
            latents = latents * 0.18215
        
        
        # recovered image tensor back
        latents_recovered = 1 / 0.18215 * latents
        recovered_depth_normalized = vae.decode(latents_recovered).sample
        
        
        
        

        
        recovered_denoise = de_normalization(resized_left_disp.squeeze(0).permute(1,2,0).cpu().numpy(),recovered_depth_normalized.squeeze(0).permute(1,2,0).cpu().numpy())

        print(np.mean(np.abs(recovered_denoise-resized_left_disp.squeeze(0).permute(1,2,0).cpu().numpy())))

        
        # print((normaliazed_left_disp-recovered_depth_normalized).mean())
        # print(normaliazed_left_disp.mean())
        # print(recovered_depth_normalized.mean())
        
        break
        
    
    
    
    
    
    
    