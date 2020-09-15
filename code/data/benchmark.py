import os

from data import common
from data import dehazebasedata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

import glob


class Benchmark(dehazebasedata.DehazeBaseData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_latent = glob.glob(os.path.join(self.dir_latent, '*' + self.ext))
        list_haze = glob.glob(os.path.join(self.dir_haze, '*' + self.ext))

        list_latent.sort()
        list_haze.sort()

        return list_latent, list_haze

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, "../../benchmark", self.args.data_test)
        if (
            self.args.data_test == "dehaze_test"
            
        ):
            self.apath = os.path.join(dir_data, "test")
        self.dir_latent = os.path.join(self.apath, "latent")
        self.dir_haze = os.path.join(self.apath, "haze")
        self.ext = ".png"
