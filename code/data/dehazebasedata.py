import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import glob

class DehazeBaseData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark

        self._set_filesystem(args.dir_data)

        if args.ext == 'img' or benchmark:
            if self.train:
                self.images_latent, self.images_A, self.images_t, self.images_haze = self._scan()
            else:
                self.images_latent, self.images_haze = self._scan()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_blurbin(self):
        raise NotImplementedError

    def _name_latentbin(self):
        raise NotImplementedError
    
    def _name_kernelbin(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.train:
            torch.manual_seed(idx)

            latent, A, t, haze, filename = self._load_file(idx)
            latent_tensor, A_tensor, t_tensor, haze_tensor = common.np2Tensor([latent, A, t, haze], self.args.rgb_range)
            A_tensor = A_tensor.view(1, A_tensor.shape[0], A_tensor.shape[1])
            t_tensor = t_tensor.view(1, t_tensor.shape[0], t_tensor.shape[1])
            haze_tensor, A_tensor, t_tensor, latent_tensor = self._get_patch(haze_tensor, A_tensor, t_tensor, latent_tensor)

            return haze_tensor, A_tensor, t_tensor, latent_tensor, filename

        else:
            torch.manual_seed(-idx)

            latent, haze, filename = self._load_file(idx)
            latent_tensor, haze_tensor = common.np2Tensor([latent, haze], self.args.rgb_range)

            return haze_tensor, latent_tensor, filename

    def __len__(self):
        return len(self.images_latent)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        
        idx = self._get_index(idx)
        
        if self.train:
            path_latent = self.images_latent[idx]
            filename_latent = os.path.splitext(os.path.split(path_latent)[-1])[0]
            path_A = self.images_A[idx]
            filename_A = os.path.splitext(os.path.split(path_A)[-1])[0]
            path_t = self.images_t[idx]
            filename_t = os.path.splitext(os.path.split(path_t)[-1])[0]
            path_haze = self.images_haze[idx]
            filename_haze = os.path.splitext(os.path.split(path_haze)[-1])[0]
            
            assert(filename_latent == filename_A)
            assert(filename_latent == filename_t)
            assert(filename_latent == filename_haze)
            filename = filename_latent
            
            # kernel = self.kernel[idx]
            if self.args.ext == 'img' or self.benchmark:

                latent = misc.imread(path_latent)
                A = misc.imread(path_A)
                t = misc.imread(path_t)
                haze = misc.imread(path_haze)
            else:
                filename = str(idx + 1)

            return latent, A, t, haze, filename

        else:
            path_latent = self.images_latent[idx]
            filename_latent = os.path.splitext(os.path.split(path_latent)[-1])[0]
            path_haze = self.images_haze[idx]
            filename_haze = os.path.splitext(os.path.split(path_haze)[-1])[0]
            
            # assert(filename_latent == filename_haze)
            filename = filename_latent
            
            # kernel = self.kernel[idx]
            if self.args.ext == 'img' or self.benchmark:

                latent = misc.imread(path_latent)
                haze = misc.imread(path_haze)
            else:
                filename = str(idx + 1)

            return latent, haze, filename

    def _get_patch(self, haze_tensor, A_tensor, t_tensor, latent_tensor):
        patch_size = self.args.patch_size
        if self.train:
            haze_tensor, A_tensor, t_tensor, latent_tensor = common.get_patch(
                haze_tensor, A_tensor, t_tensor, latent_tensor, patch_size
            )
        else:
            pass

        return haze_tensor, A_tensor, t_tensor, latent_tensor


