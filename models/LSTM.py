#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Time-Series-Library-main 
@File    : LSTM.py
@IDE     : PyCharm 
@Author  : Lifeng
@Date    : 2023/9/30 19:32 
@Software: PyCharm
'''

import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.hidden_size = configs.pred_len  # 匹配输出长度
        self.input_size = configs.enc_in
        self.num_layers = 1
        self.output_size = configs.enc_in
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                            dropout=configs.dropout)
        self.fc = nn.Linear(self.seq_len, self.output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ):
        x = x_enc
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        out = out.permute(0, 2, 1)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, :, :])


        return out
