import glob
import os
import torch
import numpy as np
from loguru import logger
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


if __name__ == "__main__":

    # D:\botao\BiFormer-mainv
    # frame1_frame2_pth = "data/raw/img_rename"
    # frame3_pth = "data/raw/gt_rename_remask"
    # img_pth = "data/train/img"
    # mask_pth = "data/train/mask"
    # frame1_pth = "data/train/frame1"
    # frame3_pth = "data/train/frame3"
    # frame2_pth = "data/train/frame2"


    # frame1_frame2_file_lst = sorted(glob.glob(str(frame1_frame2_pth) + "/*.jpg"))
    # for i in tqdm(frame1_frame2_file_lst):
    #     root = i.split(".jpg")[0]
    #     index = root.split("/")[-1]
    #     cur_img_pth = img_pth
    #     if len(index) == 4:
    #         cur_img_pth = img_pth + "/" + str(index) + ".jpg"
    #     else:
    #         cur_img_pth = img_pth + "/" +"0"*(4-len(index))+str(index) + ".jpg"
    #     f1 = Image.open(i)
    #     f1.save(cur_img_pth)
    # frame3_file_lst = glob.glob(str(frame3_pth) + "/*.jpg")
    # for i in tqdm(frame3_file_lst):
    #     root = i.split(".jpg")[0]
    #     index = root.split("/")[-1]
    #     cur_img_pth = img_pth
    #     if len(index) == 4:
    #         cur_img_pth = mask_pth + "/" + str(index) + ".jpg"
    #     else:
    #         cur_img_pth = mask_pth + "/" +"0"*(4-len(index))+str(index) + ".jpg"
    #     f1 = Image.open(i)
    #     f1.save(cur_img_pth)
    img_pth = "data/train/img"
    mask_pth = "data/train/mask"
    frame1_pth = "data/train/frame1"
    frame3_pth = "data/train/frame3"
    frame2_pth = "data/train/frame2"

    frame1_frame2_file_lst = glob.glob(str(img_pth) + "/*.jpg")
    frame3_file_lst = glob.glob(str(mask_pth) + "/*.jpg")

    if len(frame1_frame2_file_lst) != len(frame3_file_lst):
        print("The numbers of mask not equal")
    else:
        for i in tqdm(range(0,len(frame1_frame2_file_lst),3)):
            frame1_img_pth = frame1_frame2_file_lst[i]
            frame3_img_pth = frame1_frame2_file_lst[i+2]
            frame2_img_pth = frame3_file_lst[i+1]

            frame1 = Image.open(frame1_img_pth)
            frame3 = Image.open(frame3_img_pth)
            frame2 = Image.open(frame2_img_pth)

            frame1_root = frame1_img_pth.split(".jpg")[0]
            frame1_index = frame1_root.split("/")[-1]
            frame1_new_pth = frame1_pth + "/" + "0"*(4-len(frame1_index))+str(frame1_index) + ".jpg"

            # frame3_root = frame3_img_pth.split(".jpg")[0]
            # frame3_index = frame3_root.split("/")[-1]
            frame3_new_pth = frame3_pth + "/" + "0" * (4 - len(frame1_index)) + str(frame1_index) + ".jpg"

            # frame2_root = frame2_img_pth.split(".jpg")[0]
            # frame2_index = frame2_root.split("/")[-1]
            frame2_new_pth = frame2_pth + "/" + "0" * (4 - len(frame1_index)) + str(frame1_index) + ".jpg"

            frame1.save(frame1_new_pth)
            frame2.save(frame2_new_pth)
            frame3.save(frame3_new_pth)


    # To frame1 rgb

    # To frame3 rgb

    # To frame2 l
    # D:\botao\BiFormer - main\data\raw\gt_rename_remask