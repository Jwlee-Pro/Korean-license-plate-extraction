# Generate super-resolution images (latent diffusion model, LDM)
# Coded by Joonwon Lee:  2023.1.30
# Revised v2:  2023.2.17
# Output: Images with higher resolution


# os.system('pip install git+https://github.com/huggingface/diffusers.git')
# os.system('pip install accelerate')


import os, sys, time
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def super_resol_LDM(input_loc, output_loc, iternum=10, width=nan, height=nan):
    ####################################################################
    # Load SR weights from hugging face (LDM)
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    ####################################################################
    # Apply super-resolution
    if not output_loc:
        os.makedirs(output_loc, exist_ok=True)

    imgs = np.sort(glob.glob(f"{input_loc}*.jpg"))
    for ii, img in enumerate(imgs):
        im = Image.open(img)
        (x, y) = im.size

        # apply SR
        low_res_img = im.convert("RGB")
        low_res_img = low_res_img.resize((128, 128))

        # run pipeline in inference (sample random noise and denoise)
        upscaled_image = pipeline(
            low_res_img, num_inference_steps=iternum, eta=1
        ).images[0]

        # resize image
        if np.isnan(width) and np.isnan(height):
            upscaled_image = upscaled_image.resize((x * 2, y * 2))
        else:
            upscaled_image = upscaled_image.resize((width, height))

        # save image
        filename = img.split("/")[-1]
        upscaled_image.save(f"{output_loc}/{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_loc", required=True, help="path to input jpg image dataset"
    )
    parser.add_argument(
        "--output_loc", required=True, help="path to output image dataset"
    )
    parser.add_argument("--iter", type=int, default=10, help="Number of SR iteration")
    parser.add_argument("--width", type=int, default=0, help="width of output image")
    parser.add_argument("--height", type=int, default=0, help="height of output image")

    opt = parser.parse_args()

    super_resol_LDM(opt.input_loc, opt.output_loc, opt.iternum, opt.width, opt.height)
