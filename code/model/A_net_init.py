from model import common
from model import unet

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return A_net_init(args)

class A_net_init(nn.Module):
    def __init__(self, args):
        super(A_net_init, self).__init__()

        self.args = args
        m_unet = [unet.make_model(3, 1, ngf = args.n_feats)]
        self.unet = nn.Sequential(*m_unet)

    def forward(self, I):
        A = self.unet(I)
        return A

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

