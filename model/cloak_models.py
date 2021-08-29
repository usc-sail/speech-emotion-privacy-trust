#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
"""
from re import T
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
import pdb

from torch.nn.modules import dropout
import itertools


class NoisyActivation(nn.Module):
    def __init__(self,  given_locs, given_scales, min_scale, max_scale, device):
        super(NoisyActivation, self).__init__()
        size = given_scales.shape
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.given_locs = given_locs 
        self.given_scales = given_scales
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs), requires_grad=True)         
        self.rhos = nn.Parameter(torch.ones(size)-2, requires_grad=True) #-inf
        self.device = device

        # self.noise = nn.Parameter(torch.Tensor(size).normal_(mean=prior_mus, std=prior_sigmas))
        self.normal = torch.distributions.normal.Normal(0, 0.1)
        self.rhos.requires_grad = True
        self.locs.requires_grad = True
        
    def scales(self):
        # pdb.set_trace()
        return (1.0 + torch.tanh(self.rhos))/2*(self.max_scale-self.min_scale) + self.min_scale             
    
    def sample_noise(self, mask=None):
        if mask is not None:
            epsilon = self.normal.sample(self.rhos.shape).to(self.device)*mask
        else:
            epsilon = self.normal.sample(self.rhos.shape).to(self.device)
        return self.locs + self.scales() * epsilon           
                            
    def forward(self, input, mask=None):
        noise = self.sample_noise(mask)
        
        if mask is None:
            return (input) + noise
        else:
            return (input*mask) + noise


class two_d_cnn_lstm_syn(nn.Module):

    def __init__(self, original_model, noise_model):
        super(two_d_cnn_lstm_syn, self).__init__()
                                
        self.intermed = noise_model
        self.original_model = original_model

        for param in self.original_model.parameters():
            if param.requires_grad:
                param.requires_grad = False

            if isinstance(param, nn.modules.batchnorm._BatchNorm):
                param.eval()
                param.affine = False
                param.track_running_stats = False

        self.intermed.rhos.reuires_grad = True
        self.intermed.locs.reuires_grad = True
                                 
    def forward(self, input_var, global_feature=None, mask=None):
        
        x = input_var.float()
        if mask is None:
            x = self.intermed(x)
        else:
            x = self.intermed(x, mask)
        # pdb.set_trace()
        noisy = x.detach()
        
        x = self.original_model.conv(x.float())
        x = x.transpose(1, 2).contiguous()
        x_size = x.size()
        x = x.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x, h_state = self.original_model.rnn(x)
        
        if self.original_model.att is None:
            z = torch.mean(x, dim=1)
        elif self.original_model.att == 'self_att':
            # pdb.set_trace()
            att = self.original_model.att_linear1(x)
            att = self.original_model.att_pool(att)
            att = self.original_model.att_linear2(att)
            att = att.transpose(1, 2)
            
            att = torch.softmax(att, dim=2)
            z = torch.matmul(att, x)
            z = torch.mean(z, dim=1)
        
        if global_feature is not None:
            z = torch.cat((z, global_feature), 1)
        
        z = self.original_model.dense1(z)
        z = self.original_model.dense_relu1(z)
        z = self.original_model.dropout(z)

        if self.original_model.pred == 'multitask':
            preds1 = self.original_model.pred_emotion_layer(z)
            preds2 = self.original_model.pred_gender_layer(z)
            preds = (preds1, preds2)
        elif self.original_model.pred == 'emotion':
            preds = self.original_model.pred_emotion_layer(z)
        else:
            preds = self.original_model.pred_gender_layer(z)

        return preds, noisy