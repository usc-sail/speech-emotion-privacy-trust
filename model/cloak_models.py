#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tiantian
Reference from:
Mireshghallah, F., Taram, M., Jalali, A., Elthakeb, A.T.T., Tullsen, D. and Esmaeilzadeh, H., 2021, April. 
Not all features are equal: Discovering essential features for preserving prediction privacy. 
In Proceedings of the Web Conference 2021 (pp. 669-680).
"""
from re import T
import pandas as pd
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F
import pdb
from reversal_gradient import GradientReversal

from torch.nn.modules import dropout
import itertools


class cloak_noise(nn.Module):
    def __init__(self,  given_locs, given_scales, min_scale, max_scale, device):
        super(cloak_noise, self).__init__()
        size = given_scales.shape
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.given_locs = given_locs 
        self.given_scales = given_scales
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs), requires_grad=True)         
        self.rhos = nn.Parameter(torch.ones(size)-3, requires_grad=True) #-inf
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
                                 
    def forward(self, input_var, global_feature=None, mask=None, pooling=None):
        
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
            if pooling is None:
                x_size = x.size()
                z = x.reshape(-1, x_size[1]*x_size[2])
            else:
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


class two_d_cnn_lstm_syn_with_grl(nn.Module):

    def __init__(self, original_model, gender_model, noise_model, grl_lambda):
        super(two_d_cnn_lstm_syn_with_grl, self).__init__()
                                
        self.intermed = noise_model
        self.original_model = original_model
        self.gender_model = gender_model
        
        for param in self.original_model.parameters():
            if param.requires_grad:
                param.requires_grad = False

            if isinstance(param, nn.modules.batchnorm._BatchNorm):
                param.eval()
                param.affine = False
                param.track_running_stats = False
        
        # gender part
        self.gender_model.conv = nn.Sequential(GradientReversal(grl_lambda), gender_model.conv)

        self.intermed.rhos.reuires_grad = True
        self.intermed.locs.reuires_grad = True
                                 
    def forward(self, input_var, global_feature=None, mask=None, grl=False, pooling=None):
        
        x = input_var.float()
        x = self.intermed(x) if mask is None else self.intermed(x, mask)
        
        noisy = x.detach()

        # emotion part
        x1 = self.original_model.conv(x.float())
        x1 = x1.transpose(1, 2).contiguous()
        x_size = x1.size()
        x1 = x1.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x1, h_state = self.original_model.rnn(x1)
        
        if self.original_model.att is None:
            if pooling is None:
                x_size = x1.size()
                z1 = x1.reshape(-1, x_size[1]*x_size[2])
            else:
                z1 = torch.mean(x1, dim=1)
        elif self.original_model.att == 'self_att':
            att1 = self.original_model.att_linear1(x1)
            att1 = self.original_model.att_pool(att1)
            att1 = self.original_model.att_linear2(att1)
            att1 = att1.transpose(1, 2)
            
            att1 = torch.softmax(att1, dim=2)
            z1 = torch.matmul(att1, x1)
            z1 = torch.mean(z1, dim=1)
        
        if global_feature is not None:
            z1 = torch.cat((z1, global_feature), 1)
        
        z1 = self.original_model.dense1(z1)
        z1 = self.original_model.dense_relu1(z1)
        z1 = self.original_model.dropout(z1)
        preds1 = self.original_model.pred_emotion_layer(z1)

        # gender model
        x2 = self.gender_model.conv(x.float())
        x2 = x2.transpose(1, 2).contiguous()
        x_size = x2.size()
        x2 = x2.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x2, h_state = self.gender_model.rnn(x2)
        
        if self.original_model.att is None:
            if pooling is None:
                x_size = x2.size()
                z2 = x2.reshape(-1, x_size[1]*x_size[2])
            else:
                z2 = torch.mean(x2, dim=1)
        elif self.original_model.att == 'self_att':
            att2 = self.gender_model.att_linear1(x2)
            att2 = self.gender_model.att_pool(att2)
            att2 = self.gender_model.att_linear2(att2)
            att2 = att2.transpose(1, 2)
            
            att2 = torch.softmax(att2, dim=2)
            z2 = torch.matmul(att2, x2)
            z2 = torch.mean(z2, dim=1)
        
        if global_feature is not None:
            z2 = torch.cat((z2, global_feature), 1)
        
        z2 = self.gender_model.dense1(z2)
        z2 = self.gender_model.dense_relu1(z2)
        z2 = self.gender_model.dropout(z2)
        preds2 = self.gender_model.pred_gender_layer(z2)
        
        return preds1, preds2, noisy

