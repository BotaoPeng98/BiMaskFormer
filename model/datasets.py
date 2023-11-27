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


class MultiResolutionDataset:
    def __init__(self, scene_pth, device) -> None:
        super().__init__()
        self.device = device
        self.frame1_dataset, self.frame3_dataset, self.frame2_dataset, self.index_dataset = self.init_datasets(scene_pth)
        self.size = len(self.frame1_dataset)
        self.scene_pth = scene_pth

    def init_datasets(self, scene_pth: Path):
        frame1_lst = []
        frame3_lst = []

        frame2_lst = []
        index_lst = []

        if os.path.exists(scene_pth):
            frame1_file_list = glob.glob(str(scene_pth + "/frame1/*.jpg"))
            frame2_file_list = glob.glob(str(scene_pth + "/frame2/*.jpg"))

            frame3_file_list = glob.glob(str(scene_pth + "/frame3/*.jpg"))
            if len(frame1_file_list) != len(frame2_file_list) \
                    or len(frame1_file_list) != len(frame3_file_list) \
                    or len(frame3_file_list) != len(frame2_file_list):
                logger.error(f"The number of {len(frame1_file_list)} frames, {len(frame2_file_list)} frames\
                            and {len(frame3_file_list)} frames is not equal in {scene_pth}")

            pbar = tqdm(range(len(frame1_file_list)))
            for i in pbar:
                pbar.set_description("Loading views")
                frame1_img_pth = Path(frame1_file_list[i])
                frame3_img_pth = Path(frame3_file_list[i])
                frame2_img_pth = Path(frame2_file_list[i])
                f1 = Image.open(frame1_img_pth)
                f3 = Image.open(frame3_img_pth)
                f2 = Image.open(frame2_img_pth).convert("L")

                index = frame1_file_list[i].split('.')[0].split('\\')[-1]
                frame1_lst.append(f1)
                frame3_lst.append(f3)
                frame2_lst.append(f2)
                index_lst.append(index)

        else:
            logger.error(f"Scene path {scene_pth} does not exist")

        return frame1_lst, frame3_lst, frame2_lst, index_lst


    # dataloader collate_fn
    def collate(self, index):
        frame1 = self.frame1_dataset[index[0]]
        frame3 = self.frame3_dataset[index[0]]
        frame2 = self.frame2_dataset[index[0]]
        idx = self.index_dataset[index[0]]


        # texture_view = texture_view.repeat(1, 1, 1, 1)
        # mask = mask.repeat(1, 1, 1, 1)
        data = {'frame1': frame1,  # (B,C,H,W)?
                'frame3': frame3,
                'frame2': frame2,
                'index': idx}
        return data


    @property
    def dataloader(self) -> DataLoader:
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=True, num_workers=0)
        loader._data = self
        return loader
