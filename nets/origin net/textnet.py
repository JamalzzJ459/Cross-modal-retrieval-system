# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:32:33 2023

@author: ZZJ
"""

import torch
from torch import nn
from torch.nn import functional as F

class TextNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2,drop_rate = 0.1):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TextNet, self).__init__()
        self.module_name = "txt_model"

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            modules += [nn.Dropout(p = drop_rate)] if drop_rate > 0. else [nn.Identity()]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                    modules += [nn.Dropout(p = drop_rate)] if drop_rate > 0. else [nn.Identity()]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                    modules += [nn.Dropout(p = drop_rate)] if drop_rate > 0. else [nn.Identity()]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
            
        decode_modules = [nn.Linear(bit, mid_num2)]
        decode_modules += [nn.ReLU(inplace=True),nn.Linear(mid_num2,mid_num1),nn.ReLU(inplace = True),nn.Linear(mid_num1,y_dim)]
        self.fc = nn.Sequential(*modules)
        self.decode = nn.Sequential(*modules)
        #self.apply(weights_init)
        self.norm = norm

    def forward(self, x):
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out