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
import argparse


def lp_organizer(info):
    info_org = [""] * 6

    # split with any number
    fields = re.split(r"(\d+)", info)

    # 영업용 판단 ('영'으로 시작)
    if len(fields[0]) > 0:
        if fields[0][0] == "영":
            info_org[5] = "영"
            fields[0] = fields[0][1:]

    # organize information
    if len(fields) == 3:
        # motorcycle
        info_org[2] = fields[0][-1]
        temp = fields[0][:-1]
        info_org[0] = temp[0:2]
        temp = temp[2:]
        info_org[4] = temp
        info_org[3] = fields[1]
    else:
        # general cars
        info_org[0] = fields[0]
        info_org[1] = fields[1]
        info_org[2] = fields[2]
        info_org[3] = fields[3]

    return info_org


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_loc", required=True, help="path to input json image dataset"
    )
    parser.add_argument(
        "--output_loc", required=True, help="path to output pickle dataset"
    )

    opt = parser.parse_args()

    ####################################################################
    # Make dataframe with license plate information
    label = glob.glob(f"{opt.input_loc}/*.json")

    p1 = [""] * len(label)  # region
    p2 = [""] * len(label)  # three_num
    p3 = [""] * len(label)  # hangul
    p4 = [""] * len(label)  # four_num
    p5 = [""] * len(label)  # province
    p6 = [""] * len(label)  # 영업용 (='영')
    p7 = [""] * len(label)  # filename

    for ii, ilabel in enumerate(label):
        with open(ilabel) as f:
            data = json.load(f)
        info_org = lp_organizer(data["value"])
        p1[ii] = info_org[0]
        p2[ii] = info_org[1]
        p3[ii] = info_org[2]
        p4[ii] = info_org[3]
        p5[ii] = info_org[4]
        p6[ii] = info_org[5]
        p7[ii] = ilabel.split("/")[-1].split(".")[0]

    df_label = pd.DataFrame(
        {
            "region": p1,
            "three_num": p2,
            "hangul": p3,
            "four_num": p4,
            "province": p5,
            "construction": p6,
            "filename": p7,
        }
    )

    df_label.to_pickle(f"{opt.output_loc}/df_LPinfo.pkl")
