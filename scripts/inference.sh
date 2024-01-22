inference_single_image(){
input_rgb_path="/home/zliu/ECCV2024/Accelerator-Simple-Template/data_sample/kitti3d_000025.png"
output_dir="outputs"
pretrained_model_path="Bingxin/Marigold"
ensemble_size=10

cd ..
cd run

CUDA_VISIBLE_DEVICES=0 python run_inference.py \
    --input_rgb_path $input_rgb_path \
    --output_dir $output_dir \
    --pretrained_model_path $pretrained_model_path \
    --ensemble_size $ensemble_size
    }


inference_single_image_sf_rgb(){
root_path="/media/zliu/data12/dataset/KITTI/KITTI_Raw/"
output_dir="rendered_right_view"
pretrained_model_path="/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/outputs/img2img_kitti_finetune/checkpoint-50000"
ensemble_size=1
filename_list="/home/zliu/Desktop/ECCV2024/code/Diffusion/sf_double_check/Accelerator-Simple-Template/datafiles/KITTI/kitti_raw_val.txt"

cd ..
cd run

CUDA_VISIBLE_DEVICES=0 python run_inference_list.py \
    --root_path $root_path \
    --output_dir $output_dir \
    --pretrained_model_path $pretrained_model_path \
    --filename_list $filename_list
    }


inference_single_image_sf_rgb


