import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import sys
sys.path.append("../")
from Inference.image_pipeline import DepthEstimationPipeline

from utils.seed_all import seed_all
import matplotlib.pyplot as plt

from dataloader.utils import read_text_lines


from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
import skimage.io


if __name__=="__main__":
        
    use_seperate = True
    stable_diffusion_repo_path = "stabilityai/stable-diffusion-2"
    logging.basicConfig(level=logging.INFO)
    
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run MonoDepth Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='None',
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--filename_list",
        type=str,
        default='None',
        help="Given a filename list, do the right image synthetics in group."
    )

    
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="root image.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",)
    
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    
    

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    
    args = parser.parse_args()
    
    checkpoint_path = args.pretrained_model_path
    root_path = args.root_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    
    if batch_size==0:
        batch_size = 1  # set default batchsize
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # Output directories

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")
    


    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    if not use_seperate:
        pipe = DepthEstimationPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
        print("Using Completed")
    else:
        
        vae = AutoencoderKL.from_pretrained(stable_diffusion_repo_path,subfolder='vae')
        scheduler = DDIMScheduler.from_pretrained(stable_diffusion_repo_path,subfolder='scheduler')
        text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_repo_path,subfolder='text_encoder')
        tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_repo_path,subfolder='tokenizer')
        
        # https://huggingface.co/docs/diffusers/training/adapt_a_model
        unet = UNet2DConditionModel.from_pretrained(checkpoint_path,subfolder="unet",
                                                    in_channels=8, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
        
        pipe = DepthEstimationPipeline(unet=unet,
                                       vae=vae,
                                       scheduler=scheduler,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer)
        print("Using Seperated Modules")
    
    logging.info("loading pipeline whole successfully.")
    
    try:

        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    
    lines = read_text_lines(args.filename_list)
    inference_image_nums = len(lines)
    
    
    for idx, lines in enumerate(lines):
        fname= lines.split()[0]
        input_image_path = os.path.join(root_path,fname)
        
        logging.info("Processing {}/{}".format(idx+1,inference_image_nums))
        with torch.no_grad():
            # os.makedirs(output_dir, exist_ok=True)
            # load the example image.
            input_image_pil = Image.open(input_image_path)
            # input_image_pil.save("input_image.png")
            # predict the depth here
            pipe_out = pipe(input_image_pil,
                denosing_steps=denoise_steps,
                processing_res = processing_res,
                match_input_res = match_input_res,
                batch_size = batch_size,
                color_map = color_map,
                show_progress_bar = True,
                )

            rendered_rgb = pipe_out                
            save_name = os.path.basename(input_image_path)
            relative_input_image_path = input_image_path[len("/media/zliu/data12/dataset/KITTI/KITTI_Raw/"):]
            relative_input_image_path = relative_input_image_path.replace("image_02",'image_03')
            saved_path = os.path.join(output_dir,relative_input_image_path)
            saved_folder = saved_path[:-len(save_name)]
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
            
            rendered_rgb = (rendered_rgb * 255).astype(np.uint8)
            # rendered_rgb_pil= Image.fromarray(rendered_rgb)
            
            skimage.io.imsave(saved_path,rendered_rgb)
            # rendered_rgb_pil.save(saved_path)
            
            torch.cuda.empty_cache()


        
    