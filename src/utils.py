# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from logging import getLogger
import numpy as np
from .logger import create_logger, PD_Stats
import torch.nn as nn
import torch
FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

logger = getLogger()

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, shift=2., measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.shift = shift
        if measure == 'order':
            self.sim = order_sim
        else:
            #计算相似度
            self.sim = lambda x, y: x.mm(y.t())

        self.max_violation = max_violation
        self.count = 1

    def set_margin(self, margin):
        self.margin = margin

    def loss_func(self, cost, tau):
        cost = (cost - cost.diag().reshape([-1, 1])).exp()
        I = (cost.diag().diag() == 0)
        return cost[I].sum() / (cost.shape[0] * (cost.shape[0] - 1))
    def compute_loss(self,scores,B,tau):
        diagonal = scores.diag().view(B, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        #计算掩码，如果Sij>=Sii-m 则为1，表示在圈内
        mask_s = (scores >= (d1 - self.margin)).float().detach()
        #计算新的相似度得分，cost_s = scores +(1-mask_s)*shift,也就是说如果负样本离得比较远，距离大于margin了，就会让它的得分更高一点从而把它踢的更远？
        cost_s = scores * mask_s + (1. - mask_s) * (scores - self.shift)
        #同上   Sij>= Sjj-m
        mask_im = (scores >= (d2 - self.margin)).float().detach()
        cost_im = scores * mask_im + (1. - mask_im) * (scores - self.shift)
        # 希望正得越高越好，负样本得分越低越好
        loss = (-cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + self.margin).mean() + (-cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + self.margin).mean()
        return loss
    def forward(self, im, s,im2=None,s2=None, tau=0.9, lab=None):
        # compute image-sentence score matrix
        B = im.size(0)
        scores = self.sim(im, s)
        scores2 = self.sim(im2,s2)
        scores_ii = self.sim(im,s2)
        scores_tt = self.sim(s,s2)
        
        loss_it = self.compute_loss(scores,B,tau)+ self.compute_loss(scores2, B, tau)
        #loss_ii = self.compute_loss(scores_ii, B, tau)
        #loss_tt = self.compute_loss(scores_tt, B, tau)
        #loss = loss_it+ 0.5*loss_ii + 0.5*loss_tt
        loss = loss_it
        return loss
    def get_it_loss(self, img, text, C, tau=0.75,m = 0.2,mu=0.5):
        # compute image-sentence score matrix
        '''
        B = img.size(0)
        scores = self.sim(img, text)
        loss_it = self.compute_loss(scores,B,tau)
        loss = loss_it
        '''
        #mask_c = (C<(1-tau)).float()
        #print(C)
        B = img.size(0)
        scores = self.sim(img,text)/2 + 0.5
        diagonal = scores.diag().view(B, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        #计算掩码，如果Sij>=Sii-m 则为1，表示在圈内
        delta =scores - d1
        delta2 = scores -d2
        out = delta +mu*(1-C)
        out2 = delta2 +mu*(1-C)
        out = torch.relu(out)
        out2 = torch.relu(out2)
        loss_r = torch.sum(out)  +torch.sum(out2)
        loss_r2 = torch.sum(torch.relu(-(delta +mu*(1-C)))) +torch.sum(torch.relu(-(delta2 +mu*(1-C))))
        loss = loss_r  +loss_r2
        return loss

def load_image(image_paths):
    imageResolution=224
    transform = Compose([
                Resize(imageResolution, interpolation=Image.BICUBIC),
                CenterCrop(imageResolution),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
    num_images = len(image_paths)
    tensor = torch.zeros((num_images, 3, 224, 224))

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path[0].strip()).convert("RGB")
        image = transform(image)
        tensor[i] = image
    
    return tensor
    
