import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
import os
import torchvision
import torchvision.transforms as transforms
from simple_conv import ConvNet

import math
from diffusers.optimization import get_scheduler

from torch.optim.lr_scheduler import OneCycleLR

from utils import accuracy

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)



if __name__=="__main__":
    accelerator = Accelerator()
    batch_size =1
    num_classes = 10

    '''Datasets'''
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    
    '''Networks'''
    model = ConvNet(num_classes=num_classes)
    
    
    '''optimizer/sc'''
    logger.info("Leanring Superparameters")
    # scale the learning rate based on the batch size, num of GPUS and accumualted steps
    learning_rate = 1e-4
    if False:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=learning_rate)
    
    
    
    # loaded the model
    
    ckpt = torch.load("saved_path/ckpt_5.pt")
    
    model.load_state_dict(ckpt['model_state'])

    model.eval()
    model.cuda()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for i,(val_images,val_labels) in enumerate(test_loader):
            val_images = val_images.to(accelerator.device)
            val_labels = val_labels.to(accelerator.device)
            out = model(val_images)
            _, val_predicted = torch.max(out.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
    
        val_acc_cur_epoch = 100 * val_correct/val_total
        logger.info("accurate rate is {}".format(val_acc_cur_epoch),main_process_only=True)
    
    