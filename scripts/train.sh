LAUNCH_TRAINING(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/data1/liu'
dataset_name='sceneflow'
trainlist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/sceneflow/SceneFlow_With_Occ.list'
vallist='/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
output_dir='../outputs/sceneflow_fine_tune_hardest'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=4
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='sceneflow_pretrain_tracker_hardest'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"  --multi_gpu depth2image_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --enable_xformers_memory_efficient_attention \

}


LAUNCH_TRAINING_IMAGE(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/media/zliu/data12/dataset/'
dataset_name='sceneflow'
trainlist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/sceneflow/SceneFlow_With_Occ.list'
vallist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
output_dir='../outputs/img2img'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=4
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='sceneflow_pretrain_tracker_img2img'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"   image2image_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing 

}

LAUNCH_TRAINING_IMAGE_KITTI(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/media/zliu/data12/dataset/KITTI/KITTI_Raw/'
dataset_name='kitti_raw'
trainlist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_val.txt'
output_dir='../outputs/img2img_kitti'
train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=16
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_img2img'


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"   image2image_kitti_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing 

}


LAUNCH_TRAINING_IMAGE_KITTI_FineTune(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='stabilityai/stable-diffusion-2'
root_path='/media/zliu/data12/dataset/KITTI/KITTI_Raw/'
dataset_name='kitti_raw'
trainlist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_train.txt'
vallist='/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_val.txt'
output_dir='../outputs/img2img_kitti_finetune'
train_batch_size=1
num_train_epochs=30
gradient_accumulation_steps=16
learning_rate=2e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='kitti_tracker_img2img_finetune'
resume_from_checkpoint="/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/outputs/img2img_kitti/checkpoint-20000"


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision="fp16"   image2image_kitti_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --resume_from_checkpoint $resume_from_checkpoint

}


LAUNCH_TRAINING_IMAGE_KITTI_FineTune