import os
import glob
import numpy as np
import json
import re
import pandas as pd
import copy
import shutil
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import argparse


def resize_images(input_loc, output_loc, width=100, height=100):
    img1 = glob.glob(f"{input_loc}/*.jpg")
    for ii, img in enumerate(img1):
        im = Image.open(img)
        im = im.resize((width, height))
        im = im.convert("RGB")
        fname = img.split("/")[-1]
        im.save(f"{output_loc}/{fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_loc", required=True, help="path to input jpg image dataset"
    )
    parser.add_argument(
        "--output_loc", required=True, help="path to output image dataset"
    )
    parser.add_argument(
        "--width", type=int, required=True, help="width of output image"
    )
    parser.add_argument(
        "--height", type=int, required=True, help="height of output image"
    )

    opt = parser.parse_args()

    resize_images(opt.input_loc, opt.output_loc, opt.width, opt.height)
