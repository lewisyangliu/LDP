import os

from data import common
from data import dehazebasedata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

import glob

class dehazedata(dehazebasedata.DehazeBaseData):
    def __init__(self, args, train=True):
        super(dehazedata, self).__init__(args, train)
        self.repeat = 1

    def _scan(self):
        list_latent = glob.glob(os.path.join(self.dir_latent, '*' + self.ext))
        list_A = glob.glob(os.path.join(self.dir_A, '*' + self.ext))
        list_t = glob.glob(os.path.join(self.dir_t, '*' + self.ext))
        list_haze = glob.glob(os.path.join(self.dir_haze, '*' + self.ext))

        list_latent.sort()
        list_A.sort()
        list_t.sort()
        list_haze.sort()

        return list_latent, list_A, list_t, list_haze

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'train')
        self.dir_latent = os.path.join(self.apath, 'latent')
        self.dir_A = os.path.join(self.apath, 'A')
        self.dir_t = os.path.join(self.apath, 't')
        self.dir_haze = os.path.join(self.apath, 'haze')
        self.ext = '.png'

    def _name_blurbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_blur.npy'.format(self.split)
        )

    def _name_latentbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_latent.npy'.format(self.split)
        )
    
    def _name_kernelbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_kernel.npy'.format(self.split)
        )

    def __len__(self):
        if self.train:
            return len(self.images_latent) * self.repeat
        else:
            return len(self.images_latent)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_latent)
        else:
            return idx

