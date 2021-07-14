# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    # def forward(self, x):
    #     b, c, _, _ = x.size()
    #     y = self.avg_pool(x).view(b, c)
    #     y = self.fc(y).view(b, c, 1, 1)
    #     return x * y.expand_as(x)
    def forward(self, x):
        b, c, = x.size()
        y = x.view(b, c)
        y = self.fc(y)
        return x * y

class TextCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 vocab_size=0,
                 seq_length=0,
                 num_layers=5,
                 kernel_sizes=[1, 2, 3, 4, 5],
                 strides=[1, 1, 1, 1, 1],
                 paddings=[0, 0, 0, 0 ,0],
                 num_filters=[200, 200, 200, 200, 200],
                 embedding_matrix=None,
                 requires_grads=False):
        '''
        initialization
        ⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param embedding_dim: embedding dim
        :param dropout_rate: drouput rate
        :param num_class: the number of label
        :param vocab_size: vocabulary size
        :param seq_length: max length of sequence after padding
        :param num_layers: the number of cnn
        :param kernel_sizes: list of conv kernel size
        :param strides: list of conv strides
        :param paddings: list of padding
        :param num_filters: list of num filters
        :param embedding_matrix: pretrained embedding look-up table,shape is:[vocab_size,embedding_dim]
        :param requires_grads: whether to update gradient of embedding in training
        '''
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.embedding_matrix = embedding_matrix
        self.requires_grads = requires_grads

        if self.num_layers != len(self.kernel_sizes) or self.num_layers != len(self.num_filters):
            raise ValueError("The number of num_layers and num_filters must be equal to the number of kernel_sizes!")

        # embedding
        if self.embedding_matrix is None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_dim,
                                          padding_idx=0)
        else:
            print('Loading pretrained embedding...')
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=self.requires_grads)
            self.vocab_size = self.embedding_matrix.shape[0]

        # conv layers
        self.conv1ds = []
        self.global_max_pool1ds = []
        final_hidden_size = sum(self.num_filters)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[0],
                                            kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),

                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[0],
                                             kernel_size=2, stride=1, padding=0),
                                   nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),

                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[0],
                                             kernel_size=3, stride=1, padding=0),
                                   nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),

                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )

        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[0],
                                             kernel_size=4, stride=1, padding=0),
                                   nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),

                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )

        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[0],
                                             kernel_size=5, stride=1, padding=0),
                                   nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),
                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )

        # dropout
        self.fc = nn.Sequential(
            nn.Linear(final_hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, self.num_class),
        )
        # self.se = SELayer(final_hidden_size)
        # self.dropout = nn.Dropout(p=self.dropout_rate)
        # self.classifier = nn.Linear(in_features=final_hidden_size, out_features=self.num_class)

    def forward(self, input_ids):
        '''
        forward propagation
        :param inputs: [batch_size,seq_length]
        :return: [batch_size,num_class]
        '''

        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        # x = nn.Dropout(p=self.dropout_rate)(x)

        # Convolution & Pooling
        # x1 = self.global_max_pool1d(F.relu(self.conv3(x))).squeeze(dim=-1)
        # x2 = self.global_max_pool1d(F.relu(self.conv4(x))).squeeze(dim=-1)
        # x3 = self.global_max_pool1d(F.relu(self.conv5(x))).squeeze(dim=-1)
        x1 = self.conv1(x).squeeze(dim=-1)
        x2 = self.conv2(x).squeeze(dim=-1)
        x3 = self.conv3(x).squeeze(dim=-1)
        x4 = self.conv4(x).squeeze(dim=-1)
        x5 = self.conv5(x).squeeze(dim=-1)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3, x4, x5), dim=-1)
        # x = self.dropout(x)
        # outputs = self.classifier(x)
        outputs = self.fc(x)

        return outputs





