import os
import glob
import numpy as np
import json
import re
import pandas as pd
import copy
import shutil
import pickle
import argparse


def _get_label(df, inds):
    label = (
        df["construction"][inds],
        df["region"][inds],
        df["province"][inds],
        df["three_num"][inds],
        df["hangul"][inds],
        df["four_num"][inds],
    )
    return label


def _ready_necessary_files_and_folders(output_loc):
    # Create a folder
    os.makedirs(output_loc, exist_ok=True)

    # Copy/paste revised repo
    os.system(f"cp -r ../models/deep-text-recognition-benchmark/. {output_loc}")

    # Update main files
    os.system(f"cp -rf ../../src/models/train.sh {output_loc}")
    os.system(f"cp -rf ../../src/models/train.py {output_loc}")

    # Make directories
    os.makedirs(f"{output_loc}data/train", exist_ok=True)
    os.makedirs(f"{output_loc}data/valid", exist_ok=True)
    os.makedirs(f"{output_loc}data/test", exist_ok=True)
    os.makedirs(f"{output_loc}data/train/Images", exist_ok=True)
    os.makedirs(f"{output_loc}data/valid/Images", exist_ok=True)
    os.makedirs(f"{output_loc}data/test/Images", exist_ok=True)


def _separate_data(labelinfo_loc, data_ratio):
    # Separate train, validation, test set
    df_label = pd.read_pickle(labelinfo_loc)
    l = len(df_label)
    r0 = data_ratio[0]
    r1 = data_ratio[1]

    train_valid_offset = int(np.round(l * r0))
    valid_test_offset = int(np.round(l * (r0 + r1)))

    inds = np.random.permutation(l)
    inds_train = inds[0:train_valid_offset]
    inds_valid = inds[train_valid_offset:valid_test_offset]
    inds_test = inds[valid_test_offset:-1]
    return df_label, inds_train, inds_valid, inds_test


def _make_gt_txt(df, inds, output_loc, filename):
    gt_txt = ""
    for ii in np.arange(len(inds)):
        label = _get_label(df, inds[ii])
        label_out = ""
        for il in np.arange(6):
            if not (label[il]):
                continue
            else:
                label_out = label_out + " " + label[il]
        label_out = label_out[1:]
        imgLoc = f'Images/{df["filename"][inds[ii]]}.jpg'
        gt_txt += "{}\t{}\n".format(imgLoc, label_out)
    gt_txt = gt_txt.strip("\n")
    with open(f"{output_loc}/data/{filename}", "w") as f:
        print(gt_txt, file=f)


def _make_image_files(df_label, inds, input_loc, output_loc, target):
    for ii in np.arange(len(inds)):
        fname = f'{df_label["filename"][inds[ii]]}.jpg'
        bscript = f"cp -rf {input_loc}/{fname} {output_loc}/data/{target}/Images"
        os.system(bscript)


def _make_lmdb_dataset(output_loc, gt_txt_filename, target):
    pyfile_in = "../models/deep-text-recognition-benchmark/create_lmdb_dataset.py"
    bscript = f"""
    python3 {pyfile_in} 
      --inputPath {output_loc}/data/{target}/ 
      --outputPath {output_loc}/data/{target}/ 
      --gtFile {output_loc}/data/{gt_txt_filename}"""
    os.system(bscript)


def _make_processed_data(common, inputs):
    df_label, input_loc, output_loc = common
    inds, gt_txt_filename, target = inputs
    _make_gt_txt(df_label, inds, output_loc, gt_txt_filename)
    _make_image_files(df_label, inds, input_loc, output_loc, target)
    _make_lmdb_dataset(output_loc, gt_txt_filename, target)


def _remove_unused_folders(output_loc):
    os.system(f"rm -rf {output_loc}/data/train/Images")
    os.system(f"rm -rf {output_loc}/data/test/Images")
    os.system(f"rm -rf {output_loc}/data/valid/Images")


def build_dataset(input_loc, labelinfo_loc, output_loc, data_ratio=[0.8, 0.1, 0.1]):
    ####################################################################
    # Adopt Naver-OCR: 'deep-text-recognition-benchmark'

    ####################################################################
    _ready_necessary_files_and_folders(output_loc=output_loc)
    df_label, inds_train, inds_valid, inds_test = _separate_data(
        labelinfo_loc, data_ratio
    )

    common = (df_label, input_loc, output_loc)

    inputs = (inds_train, "gt_train.txt", "train")
    _make_processed_data(common, inputs)

    inputs = (inds_valid, "gt_valid.txt", "valid")
    _make_processed_data(common, inputs)

    inputs = (inds_test, "gt_test.txt", "test")
    _make_processed_data(common, inputs)

    _remove_unused_folders(output_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_loc", required=True, help="path to input jpg image dataset"
    )
    parser.add_argument(
        "--labelinfo_loc", required=True, help="path to input label pickle file"
    )
    parser.add_argument(
        "--output_loc", required=True, help="path to output mdb dataset"
    )
    parser.add_argument(
        "--data_ratio", default=[0.8, 0.1, 0.1], help="ratio of train/validation/test"
    )

    opt = parser.parse_args()

    build_dataset(opt.input_loc, opt.labelinfo_loc, opt.output_loc, opt.data_ratio)
