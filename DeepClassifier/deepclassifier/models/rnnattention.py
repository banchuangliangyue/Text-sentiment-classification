# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from deepclassifier.layers import LSTM
import torch.nn.functional as F

class RNNAtt(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 dropout_rate,
                 num_class,
                 vocab_size=0,
                 seq_length=0,
                 rnn_type="lstm",
                 num_layers=2,
                 embedding_matrix=None,
                 requires_grads=False):
        '''
        initialization
        ⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param embedding_dim: embedding dim
        :param hidden_size: hidden size of rnn
        :param dropout_rate: dropout rate
        :param num_class: the  number of  label
        :param vocab_size: vocabulary size
        :param seq_length: max length of sequence after padding
        :param rnn_type: the type of rnn, which must be lstm or gru. Default: lstm.
        :param num_layers: the number of rnn layer.Default: 1.
        :param embedding_matrix: pretrained embedding look-up table,shape is [vocab_size,embedding_dim]
        :param requires_grads: whether to update gradient of embedding look up table in training stage
        '''
        super(RNNAtt, self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.hidden_size2 = self.hidden_size//2
        self.num_class = num_class
        self.rnn_type = rnn_type
        self.dropout_rate=dropout_rate
        self.num_layers = num_layers
        self.embedding_matrix = embedding_matrix
        self.requires_grads = requires_grads

        if self.embedding_matrix is None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_dim,
                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=self.requires_grads)
            self.vocab_size = self.embedding_matrix.shape[0]

        if self.rnn_type == "lstm":
            self.birnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                 num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout_rate)
        elif self.rnn_type == "gru":
            self.birnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout_rate)
        self.tanh1 = nn.Tanh()
        self.W = nn.Parameter(torch.zeros(self.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size2)
        self.fc = nn.Linear(self.hidden_size2, self.num_class)
        # self.dropout=nn.Dropout(p=self.dropout_rate)
        # self.classifier = nn.Linear(in_features=self.hidden_size * 2, out_features=self.num_class)

    def forward(self, input_ids,input_len=None):
        '''
        forward propagation
        :param inputs: [batch_size,seq_length]
        :return: [batch_size,num_class]
        '''

        x = self.embedding(input_ids)
        H, _ = self.birnn(x)

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.W), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        outputs = self.fc(out)  # [128, 64]


        return outputs