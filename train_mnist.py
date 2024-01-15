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

import torch.utils.checkpoint
from simple_conv import ConvNet

import math
import accelerate
from diffusers.optimization import get_scheduler

from torch.optim.lr_scheduler import OneCycleLR
from packaging import version
from utils import accuracy

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

model_config = {    
    "saved_path": "saved_path2",
    "logging_path": "logs",
    'gradient_accumulated_steps': 4,
    'mix_prcsion':"fp16",
    "seed": 0,
    "scale_lr": False,
    "use_ubir_adm":False,
    "lr_warming_up_steps": 500,
    "max_steps": None,
    "tracker_name": "dog-cat",
    "saved_format": 'whole',
    "use_custom_acce_save": False
}


def main():
    
    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    
    
    '''Build The Accelearte'''
    # define the acceralte project
    accelerator_project_config = ProjectConfiguration(project_dir=model_config['saved_path'],
                                                      logging_dir=os.path.join(model_config["saved_path"],model_config['logging_path']))
    
    # define the acceleartor
    accelerator = Accelerator(gradient_accumulation_steps=model_config["gradient_accumulated_steps"],
                              mixed_precision=model_config['mix_prcsion'],
                              project_config=accelerator_project_config,
                              log_with="tensorboard"
                              )


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("-----------------------------------------------------")
    
    set_seed(model_config['seed'])
    
    '''Creating the Saving Folders'''
    if accelerator.is_main_process:
        if not os.path.exists(model_config['saved_path']):
            os.makedirs(model_config['saved_path'],exist_ok=True)
    
    
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
    model_minist = ConvNet(num_classes=num_classes)
    
    
    
    '''optimizer/sc'''
    logger.info("Leanring Superparameters")
    # scale the learning rate based on the batch size, num of GPUS and accumualted steps
    learning_rate = learning_rate * model_config['gradient_accumulated_steps'] * batch_size * accelerator.num_processes
    if model_config['use_ubir_adm']:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model_minist.parameters(),
        lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    overrode_max_train_steps = False
    num_update_per_steps = math.ceil(len(train_loader)/model_config['gradient_accumulated_steps'])
    if model_config['max_steps'] is None:
        model_config['max_steps'] = num_epochs * num_update_per_steps
        overrode_max_train_steps=True
    
    
    # using diffuser schedular
    # ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            # ' "constant", "constant_with_warmup"]
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=model_config['lr_warming_up_steps'] * accelerator.num_processes,
        num_training_steps=model_config['max_steps'] * accelerator.num_processes,
    )
    
    # # using normal schedualr
    # lr_scheduler = OneCycleLR(
    #     optimizer, max_lr=learning_rate, 
    #     epochs=num_epochs, steps_per_epoch=num_update_per_steps)
    
    logger.info('accelerator preparing...')
    model_minist, optimizer,train_loader,test_loader,lr_scheduler = accelerator.prepare(
        model_minist,optimizer,train_loader,test_loader,lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        model_config['mix_prcsion'] = accelerator.mixed_precision

    # by epoch/or by steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / model_config['gradient_accumulated_steps'])
    if overrode_max_train_steps:
        max_train_steps = num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = model_config
        accelerator.init_trackers(model_config['tracker_name'], tracker_config)
    
    
    # Train
    total_batch_size = batch_size * accelerator.num_processes * model_config['gradient_accumulated_steps']
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {model_config['gradient_accumulated_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    
    initial_global_step = global_step
    
    progress_bar = tqdm.tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch,num_train_epochs):
        model_minist.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # add this line
            with accelerator.accumulate(model_minist):
                
                outputs = model_minist(images)
                loss = criterion(outputs, labels)
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item() / model_config['gradient_accumulated_steps']
                
                # if accelerator.is_main_process:
                #     # get the step acc:
                acc1 = accuracy(outputs, labels, topk=(1, ))
                    
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # if args.use_ema:
                #     ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],'average_loss':avg_loss.detach().item(),
                    "acc_rate":acc1[0].detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >=max_train_steps:
                break
            
        # for each epoch, we shold save one and get loss
        logger.info('=' * 10 + 'Start evaluating' + '=' * 10 + "Epoch {}".format(epoch), main_process_only=True)
        model_minist.eval()
        validation_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), 
                         disable=(not accelerator.is_local_main_process))
        
        val_correct = 0
        val_total = 0
        for i,(val_images,val_labels) in validation_bar:
            with torch.no_grad():
                out = model_minist(val_images)
                _, val_predicted = torch.max(out.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_acc_cur_epoch = 100 * val_correct/val_total
        logger.info("Epoch {}'s accurate rate is {}".format(epoch,val_acc_cur_epoch),main_process_only=True)

        # save accelerate state
        accelerator.wait_for_everyone()
        # save together
        if model_config['saved_format']=='whole':
            if accelerator.is_main_process:
                save_path = os.path.join(model_config['saved_path'], f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}",main_process_only=True)
        
        
        # save seperately
        else:
            if accelerator.is_main_process:
                unwarped_model = accelerator.unwrap_model(model_minist)
                unwarped_optimizer = accelerator.unwrap_model(optimizer)
                unwarped_lr = accelerator.unwrap_model(lr_scheduler)

                torch.save(
                    {"model_state":unwarped_model.state_dict(),
                     'optim_state': unwarped_optimizer.state_dict(),
                     'lr_state': unwarped_lr.state_dict()
                     },
                    os.path.join(model_config['saved_path'],f"ckpt_{epoch+1}.pt")
                    # model_config['saved_path'] + f'ckpt_{epoch+1}.pt'
                )
                logger.info(f'checkpoint ckpt_{epoch+1}.pt is saved...')
        
    
    accelerator.end_training()
        

if __name__=="__main__":
    main()