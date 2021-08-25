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


class one_d_cnn_lstm_att(nn.Module):
    def __init__(self, input_channel, input_spec_size, cnn_filter_size, lstm_hidden_size=128, num_layers_lstm=2, pred='emotion',
                 bidirectional=True, rnn_cell='gru', attention_size=256, variable_lengths=False, global_feature=1):

        super(one_d_cnn_lstm_att, self).__init__()
        self.input_channel = input_channel
        self.input_spec_size = input_spec_size
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.num_layers_lstm = num_layers_lstm
        self.dropout_p = 0.2
        self.variable_lengths = variable_lengths
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.cnn_filter_size = cnn_filter_size
        self.rnn_input_size = self.cnn_filter_size
        self.attention_size = attention_size
        self.pred = pred

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv1d(input_spec_size, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
        )

        self.rnn = self.rnn_cell(input_size=self.rnn_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.num_layers_lstm, batch_first=True,
                                 dropout=self.dropout_p, bidirectional=self.bidirectional)
        
        d_att, n_att = self.attention_size, 8
        self.att_linear1 = nn.Linear(self.lstm_hidden_size*2, d_att)
        self.att_pool = nn.Tanh()
        self.att_linear2 = nn.Linear(d_att, n_att)
        
        self.att_mat1 = torch.nn.Parameter(torch.rand(d_att, self.lstm_hidden_size*2), requires_grad=True)
        self.att_mat2 = torch.nn.Parameter(torch.rand(n_att, d_att), requires_grad=True)

        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense2 = nn.Linear(128, 64)
        self.dense1 = nn.Linear(self.lstm_hidden_size*2+88, 128) if global_feature is 1 else nn.Linear(self.lstm_hidden_size*2, 128)
        self.pred_emotion_layer = nn.Linear(64, self.num_emo_classes) 
        self.pred_gender_layer = nn.Linear(64, self.num_gender_class)
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.squeeze(dim=1)  # (B, T, D)
        x = x.permute(0, 2, 1)
        x = self.conv(x.float())

        x = x.permute(0, 2, 1)
        x_size = x.size()
        
        # pdb.set_trace()
        x, h_state = self.rnn(x)
        # x = self.rnn_norm(x)
        
        att = self.att_linear1(x)
        att = self.att_pool(att)
        att = self.att_linear2(att)
        att = att.transpose(1, 2)
        
        att = torch.softmax(att, dim=2)
        z = torch.matmul(att, x)
        z = torch.mean(z, dim=1)
        if global_feature is not None:
            z = torch.cat((z, global_feature), 1)
        
        z = self.dense1(z)
        z = self.dense_relu1(z)
        z = self.dropout(z)
        z = self.dense2(z)
        z = self.dense_relu2(z)
        z = self.dropout(z)

        if self.pred == 'multitask':
            preds1 = self.pred_emotion_layer(z)
            preds2 = self.pred_gender_layer(z)
            preds = (preds1, preds2)
        elif self.pred == 'emotion':
            preds = self.pred_emotion_layer(z)
        else:
            preds = self.pred_gender_layer(z)

        return preds

