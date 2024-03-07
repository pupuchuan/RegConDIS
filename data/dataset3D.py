from os.path import join
import os
import random
import torch
import torch.utils.data as data
from util.Nii_utils import NiiDataRead
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pickle
import torchvision.transforms as transforms
from util.util import *


def randomcrop_Npatch(crop_size, crop_Npatch, source, target, mask):
    this_frame = crop_size
    img = source

    non_zero_z, non_zero_x, non_zero_y = np.where(mask == 1)
    non_zero_num = non_zero_x.shape[0]

    patch_index = random.sample(range(0, non_zero_num), crop_Npatch)
    patch_source = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_target = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.float32)
    patch_mask = np.zeros([crop_Npatch, this_frame[0], this_frame[1], this_frame[2]]).astype(np.int)

    for idx in range(crop_Npatch):
        z_med = non_zero_z[patch_index[idx]]
        x_med = non_zero_x[patch_index[idx]]
        y_med = non_zero_y[patch_index[idx]]
        z_frame_size = int(this_frame[0] / 2)
        x_frame_size = int(this_frame[1] / 2)
        y_frame_size = int(this_frame[2] / 2)

        if z_med < z_frame_size:
            z_this_min = 0
            z_this_max = z_frame_size * 2
        elif z_med + z_frame_size > img.shape[0]:
            z_this_max = img.shape[0]
            z_this_min = z_this_max - this_frame[0]
        else:
            z_this_min = z_med - z_frame_size
            z_this_max = z_med + z_frame_size

        if x_med < x_frame_size:
            x_this_min = 0
            x_this_max = x_frame_size * 2
        elif x_med + x_frame_size > img.shape[1]:
            x_this_max = img.shape[1]
            x_this_min = x_this_max - this_frame[1]
        else:
            x_this_min = x_med - x_frame_size
            x_this_max = x_med + x_frame_size

        if y_med < y_frame_size:
            y_this_min = 0
            y_this_max = y_frame_size * 2
        elif y_med + y_frame_size > img.shape[2]:
            y_this_max = img.shape[2]
            y_this_min = y_this_max - this_frame[2]
        else:
            y_this_min = y_med - y_frame_size
            y_this_max = y_med + y_frame_size

        patch_source[idx, :, :, :] = source[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]
        patch_target[idx, :, :, :] = target[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]
        patch_mask[idx, :, :, :] = mask[z_this_min: z_this_max, x_this_min: x_this_max, y_this_min: y_this_max]

    return np.ascontiguousarray(patch_source), np.ascontiguousarray(patch_target), np.ascontiguousarray(patch_mask)


class DatasetFromFolder_train(data.Dataset):
    def __init__(self, opt):
        self.image_dir = opt.image_dir
        self.CT_max = opt.CT_max
        self.CT_min = opt.CT_min
        self.image_filenames = os.listdir(os.path.join(self.image_dir, 'train'))
        self.crop_size = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.crop_Npatch = opt.Npatch
        self.all_patch_num = self.crop_Npatch * len(self.image_filenames)

    def __getitem__(self, index):
        this_index = int(index // self.crop_Npatch)
        this_sub = self.image_filenames[this_index]
        self.ran_num = 1

        source, spacing, origin, direction = NiiDataRead(
            join(self.image_dir, 'train', this_sub, 'Arterial.nii.gz'))
        target, spacing1, origin1, direction1 = NiiDataRead(
            join(self.image_dir, 'train', this_sub, 'NC.nii.gz'))
        mask, spacing, origin, direction = NiiDataRead(
            join(self.image_dir, 'train', this_sub, 'mask.nii.gz'))

        ct_max = self.CT_max
        ct_min = self.CT_min

        source[source < ct_min] = ct_min
        source[source > ct_max] = ct_max
        source[mask == 0] = ct_min
        source = normalization_ct(source, ct_min, ct_max)


        target[target < ct_min] = ct_min
        target[target > ct_max] = ct_max
        target[mask == 0] = ct_min
        target = normalization_ct(target, ct_min, ct_max)

        source_patch, target_patch, mask_patch = randomcrop_Npatch(self.crop_size, self.ran_num, source, target, mask)

        source = torch.tensor(source_patch).float()
        target = torch.tensor(target_patch).float()
        mask = torch.tensor(mask_patch).float()

        p1 = np.random.choice([0, 1])
        p2 = np.random.choice([0, 1])
        self.trans = transforms.Compose([
                                  transforms.RandomHorizontalFlip(p1),
                                  transforms.RandomVerticalFlip(p2),
                                       ])
        source = self.trans(source)
        target = self.trans(target)
        mask = self.trans(mask)

        return {
            'A': source,
            'B': target,
            'mask': mask
        }

    def __len__(self):
        return self.all_patch_num



