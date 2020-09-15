from model import common
from model import unet

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return t_net(args)

class t_net(nn.Module):
    def __init__(self, args):
        super(t_net, self).__init__()

        self.args = args
        m_unet = [unet.make_model(1, 1, ngf = args.n_feats)]
        self.unet = nn.Sequential(*m_unet)

    def forward(self, I_A_t_J):
        I, A, t, J = I_A_t_J[:,0:3,:,:], I_A_t_J[:,3:4,:,:], I_A_t_J[:,4:5,:,:], I_A_t_J[:,5:8,:,:]
        delta_f = (J * t + A * (1.0 - t) - I) * (J - A)
        delta_g = ((I - A)/torch.clamp(t, self.args.t_clamp) + A - J) * (-(I - A)/torch.pow(torch.clamp(t, self.args.t_clamp),2))
        t_prior = self.unet(t)
        t = t - self.args.weight_t_prior * (torch.mean(delta_f, 1, keepdim=True) + torch.mean(delta_g, 1, keepdim=True) + t_prior)
        return t

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

