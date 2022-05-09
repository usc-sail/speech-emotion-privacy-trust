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


class one_d_cnn_lstm(nn.Module):
    def __init__(self, input_channel, input_spec_size, cnn_filter_size, lstm_hidden_size=128, num_layers_lstm=2, pred='emotion',
                 bidirectional=True, rnn_cell='gru', attention_size=256, variable_lengths=False, global_feature=1, att=None):

        super(one_d_cnn_lstm, self).__init__()
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
        self.rnn_input_size = 512
        self.attention_size = attention_size
        self.pred = pred
        self.att = att

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv1d(input_spec_size, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
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

        self.classifier = nn.Sequential(
            nn.Linear(512*4, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )

        self.dense2 = nn.Linear(128, 64)
        # self.dense1 = nn.Linear(self.lstm_hidden_size*2+88, 128) if global_feature is 1 else nn.Linear(self.lstm_hidden_size*2*int(200/8), 128)
        self.dense1 = nn.Linear(self.lstm_hidden_size*2+88, 128) if global_feature is 1 else nn.Linear(512*4, 128)
        self.pred_emotion_layer = nn.Linear(128, self.num_emo_classes) 
        self.pred_gender_layer = nn.Linear(128, self.num_gender_class)
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

        # pdb.set_trace()
        x = x.permute(0, 2, 1)
        # x, h_state = self.rnn(x)
        
        if self.att is None:
            x_size = x.size()
            z = x.reshape(-1, x_size[1]*x_size[2])
            # pdb.set_trace()
            # z = torch.mean(x, dim=1)
        elif self.att == 'self_att':
            att = self.att_linear1(x)
            att = self.att_pool(att)
            att = self.att_linear2(att)
            att = att.transpose(1, 2)
            
            att = torch.softmax(att, dim=2)
            z = torch.matmul(att, x)
            z = torch.mean(z, dim=1)
        
        if global_feature is not None:
            z = torch.cat((z, global_feature), 1)
        
        z = self.classifier(z)
        
        if self.pred == 'multitask':
            preds1 = self.pred_emotion_layer(z)
            preds2 = self.pred_gender_layer(z)
            preds = (preds1, preds2)
        elif self.pred == 'emotion':
            preds = self.pred_emotion_layer(z)
        else:
            preds = self.pred_gender_layer(z)

        return preds


class two_d_cnn_lstm(nn.Module):
    def __init__(self, input_channel, input_spec_size, cnn_filter_size, lstm_hidden_size=128, num_layers_lstm=2, pred='emotion',
                 bidirectional=True, rnn_cell='gru', attention_size=256, variable_lengths=False, global_feature=1, att=None):

        super(two_d_cnn_lstm, self).__init__()
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
        self.attention_size = attention_size
        self.pred = pred
        self.att = att
        self.rnn_input_size = int(128 * input_spec_size / 8)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),
        )

        self.rnn = self.rnn_cell(input_size=self.rnn_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.num_layers_lstm, batch_first=True,
                                 dropout=self.dropout_p, bidirectional=self.bidirectional)


        d_att, n_att = self.attention_size, 16
        self.att_linear1 = nn.Linear(self.lstm_hidden_size*2, d_att, bias=False)
        self.att_pool = nn.Tanh()
        self.att_linear2 = nn.Linear(d_att, n_att, bias=False)
        
        self.att_mat1 = torch.nn.Parameter(torch.rand(d_att, self.lstm_hidden_size*2), requires_grad=True)
        self.att_mat2 = torch.nn.Parameter(torch.rand(n_att, d_att), requires_grad=True)

        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense2 = nn.Linear(128, 64)
        self.dense1 = nn.Linear(self.lstm_hidden_size*2+88, 128) if global_feature is 1 else nn.Linear(self.lstm_hidden_size*2, 128)
        self.pred_emotion_layer = nn.Linear(128, self.num_emo_classes) 
        self.pred_gender_layer = nn.Linear(128, self.num_gender_class)
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.float()
        x = self.conv(x.float())
        x = x.transpose(1, 2).contiguous()
        x_size = x.size()
        x = x.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x, h_state = self.rnn(x)
        
        if self.att is None:
            z = torch.mean(x, dim=1)
        elif self.att == 'self_att':
            # pdb.set_trace()
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

        if self.pred == 'multitask':
            preds1 = self.pred_emotion_layer(z)
            preds2 = self.pred_gender_layer(z)
            preds = (preds1, preds2)
        elif self.pred == 'emotion':
            preds = self.pred_emotion_layer(z)
        else:
            preds = self.pred_gender_layer(z)

        return preds



class deep_two_d_cnn_lstm(nn.Module):
    def __init__(self, input_channel, input_spec_size, cnn_filter_size, lstm_hidden_size=128, num_layers_lstm=2, pred='emotion',
                 bidirectional=True, rnn_cell='gru', attention_size=256, variable_lengths=False, global_feature=1, att=None):

        super(deep_two_d_cnn_lstm, self).__init__()
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
        self.attention_size = attention_size
        self.pred = pred
        self.att = att
        self.rnn_input_size = int(128 * input_spec_size / 8)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p)
        )

        self.rnn = self.rnn_cell(input_size=self.rnn_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.num_layers_lstm, batch_first=True,
                                 dropout=self.dropout_p, bidirectional=self.bidirectional)


        d_att, n_att = self.attention_size, 16
        self.att_linear1 = nn.Linear(self.lstm_hidden_size*2, d_att, bias=False)
        self.att_pool = nn.Tanh()
        self.att_linear2 = nn.Linear(d_att, n_att, bias=False)
        
        self.att_mat1 = torch.nn.Parameter(torch.rand(d_att, self.lstm_hidden_size*2), requires_grad=True)
        self.att_mat2 = torch.nn.Parameter(torch.rand(n_att, d_att), requires_grad=True)

        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense2 = nn.Linear(128, 64)
        self.dense1 = nn.Linear(self.lstm_hidden_size*2+88, 128) if global_feature is 1 else nn.Linear(self.lstm_hidden_size*2*25, 128)
        self.pred_emotion_layer = nn.Linear(128, self.num_emo_classes) 
        self.pred_gender_layer = nn.Linear(128, self.num_gender_class)
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.float()
        x = self.conv(x.float())
        x = x.transpose(1, 2).contiguous()
        x_size = x.size()
        x = x.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x, h_state = self.rnn(x)
        if self.att is None:
            x_size = x.size()
            z = x.reshape(-1, x_size[1]*x_size[2])
        elif self.att == 'self_att':
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

        if self.pred == 'multitask':
            preds1 = self.pred_emotion_layer(z)
            preds2 = self.pred_gender_layer(z)
            preds = (preds1, preds2)
        elif self.pred == 'emotion':
            preds = self.pred_emotion_layer(z)
        else:
            preds = self.pred_gender_layer(z)

        return preds


class deep_two_d_cnn_lstm_tmp(nn.Module):
    def __init__(self, input_channel, input_spec_size, cnn_filter_size, lstm_hidden_size=128, num_layers_lstm=2, pred='emotion',
                 bidirectional=True, rnn_cell='lstm', attention_size=256, variable_lengths=False, global_feature=1, att=None):

        super(deep_two_d_cnn_lstm_tmp, self).__init__()
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
        self.attention_size = attention_size
        self.pred = pred
        self.att = att
        self.rnn_input_size = int(128 * input_spec_size / 8)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p)
        )

        self.rnn = self.rnn_cell(input_size=self.rnn_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.num_layers_lstm, batch_first=True,
                                 dropout=self.dropout_p, bidirectional=self.bidirectional)


        d_att, n_att = self.attention_size, 16
        self.att_linear1 = nn.Linear(self.lstm_hidden_size*2, d_att, bias=False)
        self.att_pool = nn.Tanh()
        self.att_linear2 = nn.Linear(d_att, n_att, bias=False)
        
        self.att_mat1 = torch.nn.Parameter(torch.rand(d_att, self.lstm_hidden_size*2), requires_grad=True)
        self.att_mat2 = torch.nn.Parameter(torch.rand(n_att, d_att), requires_grad=True)

        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense2 = nn.Linear(128, 64)
        self.dense1 = nn.Linear(self.lstm_hidden_size*2+88, 128) if global_feature is 1 else nn.Linear(self.lstm_hidden_size*2*25, 128)
        self.pred_emotion_layer = nn.Linear(128, self.num_emo_classes) 
        self.pred_gender_layer = nn.Linear(128, self.num_gender_class)
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.float()
        x = self.conv(x.float())
        x = x.transpose(1, 2).contiguous()
        x_size = x.size()
        x = x.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x, h_state = self.rnn(x)
        if self.att is None:
            x_size = x.size()
            z = x.reshape(-1, x_size[1]*x_size[2])
        elif self.att == 'self_att':
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

        if self.pred == 'multitask':
            preds1 = self.pred_emotion_layer(z)
            preds2 = self.pred_gender_layer(z)
            preds = (preds1, preds2)
        elif self.pred == 'emotion':
            preds = self.pred_emotion_layer(z)
        else:
            preds = self.pred_gender_layer(z)

        return preds


class two_d_cnn(nn.Module):
    def __init__(self, input_channel, input_spec_size, cnn_filter_size, pred='emotion', 
                 global_feature=1, att=None):

        super(two_d_cnn, self).__init__()
        self.input_channel = input_channel
        self.input_spec_size = input_spec_size
        self.dropout_p = 0.5
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.cnn_filter_size = cnn_filter_size
        self.pred = pred
        self.rnn_input_size = int(64 * input_spec_size / 8)

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(32, 48, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),
            
            nn.Conv2d(48, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(self.dropout_p),
        )

        self.dense_relu1 = nn.ReLU()
        self.dense_relu2 = nn.ReLU()

        self.dense2 = nn.Linear(128, 64)
        self.dense1 = nn.Linear(1*2+88, 128) if global_feature is 1 else nn.Linear(1*2, 128)
        self.pred_emotion_layer = nn.Linear(128, self.num_emo_classes) 
        self.pred_gender_layer = nn.Linear(128, self.num_gender_class)
        self.init_weight()

        self.w1 = torch.nn.Parameter(torch.rand(50, 4), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand(50, 2), requires_grad=True)

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, input_var, global_feature=None):

        x = input_var.float()
        x = self.conv(x.float())

        x = x.transpose(1, 2).contiguous()
        x_size = x.size()
        x = x.reshape(-1, x_size[1], x_size[2]*x_size[3])
        x = x.transpose(1, 2).contiguous()
        
        if self.pred == 'emotion':
            x = torch.matmul(x, self.w1)
        else:
            x = torch.matmul(x, self.w2)
        preds = torch.mean(x, dim=1)
        return preds

