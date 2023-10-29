import math
import os
import torch
from PIL.Image import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from ecan.utils.data_utils import rgb2ycbcr
from ecan.entity.config_entity import TrainingConfig
from torchvision import transforms


class DataPreparation(Dataset):
    def __init__(self, config: TrainingConfig):
        super(DataPreparation).__init__()
        self.config = config
        self.img_list = os.listdir(str(self.config.training_data) + '/lr')
        self.totensor = transforms.ToTensor()

    def __getitem__(self, idx):
        HR = []
        LR = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            img_HR = Image.open(str(self.config.training_data) + '/hr' +
                                str(idx_frame + 1).rjust(2, '0') + '.png')
            img_LR = Image.open(
                str(self.config.training_data) + '/lr/' + str(idx_frame + 1).rjust(2, '0') + '.png')

            img_HR = np.array(img_HR, dtype=np.float32) / 255.0
            if idx_frame == idx:
                h, w, c = img_HR.shape
                SR_buicbic = np.array(img_LR.resize(
                    (w, h), Image.BICUBIC), dtype=np.float32) / 255.0
                SR_buicbic = rgb2ycbcr(
                    SR_buicbic, only_y=False).transpose(2, 0, 1)

            img_LR = np.array(img_LR, dtype=np.float32) / 255.0

            if self.inType == 'y':
                img_HR = rgb2ycbcr(img_HR, only_y=True)[np.newaxis, :]
                img_LR = rgb2ycbcr(img_LR, only_y=True)[np.newaxis, :]
            if self.inType == 'RGB':
                img_HR = img_HR.transpose(2, 0, 1)
                img_LR = img_LR.transpose(2, 0, 1)

            HR.append(img_HR)
            LR.append(img_LR)

        HR = np.stack(HR, 1)
        LR = np.stack(LR, 1)

        C, N, H, W = HR.shape
        H = math.floor(H / self.config.params_scale_factor / 4) * self.config.params_scale_factor * 4
        W = math.floor(W / self.config.params_scale_factor / 4) * self.config.params_scale_factor * 4
        HR = HR[:, :, :H, :W]
        SR_buicbic = SR_buicbic[:, :H, :W]
        LR = LR[:, :, :H // self.config.params_scale_factor, :W // self.config.params_scale_factor]

        HR = torch.from_numpy(np.ascontiguousarray(HR))
        LR = torch.from_numpy(np.ascontiguousarray(LR))
        SR_buicbic = torch.from_numpy(np.ascontiguousarray(SR_buicbic))
        return LR, HR, SR_buicbic

    def __len__(self):
        return len(self.img_list)
