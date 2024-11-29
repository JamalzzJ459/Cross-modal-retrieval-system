# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:02:01 2024

@author: ZZJ
"""

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder_img,base_encoder_txt, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q_img =  base_encoder_img(y_dim=4096, bit=dim, hiden_layer=3).cuda()
        self.encoder_k_img =  base_encoder_img(y_dim=4096, bit=dim, hiden_layer=3).cuda()
        
        self.encoder_q_txt =  base_encoder_txt(y_dim=1386, bit=dim, hiden_layer=2).cuda()
        self.encoder_k_txt =  base_encoder_txt(y_dim=1386, bit=dim, hiden_layer=2).cuda()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q_img.parameters(), self.encoder_k_img.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        for param_q, param_k in zip(
            self.encoder_q_txt.parameters(), self.encoder_k_txt.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q_img.parameters(), self.encoder_k_img.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
            
        for param_q, param_k in zip(
            self.encoder_q_txt.parameters(), self.encoder_k_txt.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_img,keys_txt):
        # gather keys before updating queue

        batch_size = keys_img.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = (keys_img.T +keys_txt.T)/2 
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k,txt_q,txt_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key image
            tx_q: a batch of query texts
            im_k: a batch of key image
        Output:
            logits, targets
        """

        # compute query features
        q_i = self.encoder_q_img(im_q)  # queries: NxC
        q_t = self.encoder_q_txt(txt_q)
        #q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder


            k_i = self.encoder_k_img(im_k)  # keys: NxC
            k_t = self.encoder_k_txt(txt_k)
            #k = nn.functional.normalize(k, dim=1)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_ii = torch.einsum("nc,nc->n", [q_i, k_i]).unsqueeze(-1)
        l_pos_tt = torch.einsum("nc,nc->n", [q_t, k_t]).unsqueeze(-1)
        l_pos_it = torch.einsum("nc,nc->n", [q_i, k_t]).unsqueeze(-1)
        l_pos_ti = torch.einsum("nc,nc->n", [q_t, k_i]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ii = torch.einsum("nc,ck->nk", [q_i, self.queue.clone().detach()])
        l_neg_tt = torch.einsum("nc,ck->nk", [q_t, self.queue.clone().detach()])
        l_neg_it = torch.einsum("nc,ck->nk", [q_i, self.queue.clone().detach()])
        l_neg_ti = torch.einsum("nc,ck->nk", [q_t, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits_ii = torch.cat([l_pos_ii,l_neg_ii], dim=1)
        logits_tt = torch.cat([l_pos_tt,l_neg_tt], dim=1)
        logits_it = torch.cat([l_pos_it,l_neg_it], dim=1)
        logits_ti = torch.cat([l_pos_ti,l_neg_ti], dim=1)
        # apply temperature
        logits_ii /= self.T
        logits_tt /= self.T
        logits_it /= self.T
        logits_ti /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits_ii.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_i,k_t)

        return logits_ii,logits_tt,logits_it,logits_ti ,labels
    def encode_img(self,img):
        img_fea = self.encoder_q_img(img)
        return img_fea
    
    def encode_txt(self,txt):
        txt_fea = self.encoder_q_txt(txt)
        return txt_fea
    def encode(self,img,txt):
        img_fea = self.encode_img(img)
        txt_fea = self.encode_txt(txt)
        return img_fea,txt_fea
