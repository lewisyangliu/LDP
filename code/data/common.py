import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

def get_patch(haze_tensor, A_tensor, t_tensor, latent_tensor, patch_size):
    assert haze_tensor.shape[1:] == A_tensor.shape[1:]
    assert haze_tensor.shape[1:] == t_tensor.shape[1:]
    assert haze_tensor.shape[1:] == latent_tensor.shape[1:]

    ih, iw = haze_tensor.shape[1:]

    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)

    haze_tensor = haze_tensor[:, iy:iy + patch_size, ix:ix + patch_size]
    A_tensor = A_tensor[:, iy:iy + patch_size, ix:ix + patch_size]
    t_tensor = t_tensor[:, iy:iy + patch_size, ix:ix + patch_size]
    latent_tensor = latent_tensor[:, iy:iy + patch_size, ix:ix + patch_size]
    
    return haze_tensor, A_tensor, t_tensor, latent_tensor

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        if img.ndim == 3:
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)
        elif img.ndim == 2:
            tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
            tensor.mul_(rgb_range / 255)
        else:
            pass
        return tensor

    return [_np2Tensor(_l) for _l in l]

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]
