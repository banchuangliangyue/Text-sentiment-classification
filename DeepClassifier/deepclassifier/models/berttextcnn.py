# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import torch
import torch.nn as nn
from transformers import *
from itertools import repeat

class BertTextCNN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 bert_path,
                 num_layers=3,
                 bert_layers=3,
                 kernel_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 strides=[1, 1, 1],
                 paddings=[0, 0, 0],
                 requires_grads=False):
        '''
        initialization
        ⚠⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param dropout_rate: dropout rate
        :param num_class: the number of label
        :param bert_path: bert config path
        :param embedding_dim: embedding dim
        :param num_layers: the number of cnn layer
        :param kernel_sizes: list of conv kernel size
        :param num_filters: list of conv filters
        :param strides: list of conv strides
        :param paddings: list of conv padding
        :param requires_grads: whther to update gradient of Bert in training stage
        '''
        super(BertTextCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bert_layers = bert_layers
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.strides = strides
        self.paddings = paddings
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.bert_path=bert_path
        self.requires_grads=requires_grads
        self.spatial_drop = SpatialDropout(0.5)

        # self.bert = AutoModel.from_pretrained(self.bert_path)
        self.bert = AutoModel.from_pretrained("bert-base-uncased",num_hidden_layers=self.bert_layers)
        if self.requires_grads is False:
            for p in self.bert.parameters():
                p.requires_grads = False

        if self.num_layers != len(self.kernel_sizes) or self.num_layers != len(self.num_filters):
            raise Exception("The number of num_layers and num_filters must be equal to the number of kernel_sizes!")

        final_hidden_size = sum(self.num_filters)


        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[0],
                                             kernel_size=self.kernel_sizes[0], stride=self.strides[0],
                                             padding=self.paddings[0]),
                                   # nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),
                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[1],
                                             kernel_size=self.kernel_sizes[1], stride=self.strides[1],
                                             padding=self.paddings[1]),
                                   # nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),
                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters[2],
                                             kernel_size=self.kernel_sizes[2], stride=self.strides[2],
                                             padding=self.paddings[2]),
                                   # nn.BatchNorm1d(self.num_filters[0]),
                                   nn.ReLU(inplace=True),
                                   nn.AdaptiveMaxPool1d(output_size=1)
                                   )


        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(in_features=final_hidden_size, out_features=self.num_class)
        self.fc = nn.Linear(in_features=self.embedding_dim, out_features=self.num_class)

    def forward(self, input_ids, attention_mask=None):
        '''
        forard propagation
        :param params: input_ids:[batch_size,max_length]
        :param params: attention_mask:[batch_size,max_length]
        :return: logits:[batch_size,num_class]
        '''

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state ##[batch_size, sequence_length, hidden_size]
        # x = bert_output.pooler_output ##[batch_size, hidden_size]

        x = x.permute(0, 2, 1)
        x1 = self.conv1(x).squeeze(dim=-1)
        x2 = self.conv2(x).squeeze(dim=-1)
        x3 = self.conv3(x).squeeze(dim=-1)
        x = torch.cat((x1, x2, x3), dim=-1)
        # x = self.spatial_drop(x)
        x = self.dropout(x)
        outputs = self.classifier(x)

        return outputs


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


