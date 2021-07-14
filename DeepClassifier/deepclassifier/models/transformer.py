# -*- coding:utf-8 -*-
'''

Author:
    Zichao Li,2843656167@qq.com

'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Transformer(nn.Module):
    def __init__(self,
                 embedding_dim,
                 dropout_rate,
                 num_class,
                 vocab_size=400000,
                 seq_length=0,
                 num_encoder=2,
                 embedding_matrix=None,
                 requires_grads=False):
        '''
        initialization
        ⚠️In default,the way to initialize embedding is loading pretrained embedding look-up table!
        :param embedding_dim: embedding dim
        :param num_class: the number of label
        :param dropout_rate: dropout rate
        :param vocab_size: vocabulary size
        :param seq_length: max length of sequence after padding
        :param num_blocks: the number of block in DPCNN model
        :param num_filters: the number of filters of conv kernel
        :param kernel_sizes: conv kernel size
        :param embedding_matrix: pretrained embedding look up table
        :param requires_grads: whether to update gradient of embedding in training stage
        '''
        super(Transformer , self).__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.embedding_matrix = embedding_matrix
        self.requires_grads = requires_grads
        self.pad_size = 50
        self.dim_model = embedding_dim
        self.num_head = 5
        self.hidden = 512
        self.num_encoder = num_encoder

        # embedding
        if self.embedding_matrix is None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                          embedding_dim=self.embedding_dim,
                                          padding_idx=0)
        else:
            print('Using glove embedding....')
            self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix, freeze=self.requires_grads)
            self.vocab_size = self.embedding_matrix.shape[0]

        self.postion_embedding = Positional_Encoding(self.embedding_dim, self.pad_size, self.dropout_rate)
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, self.dropout_rate)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(self.num_encoder)])

        self.fc1 = nn.Linear(self.pad_size * self.dim_model, self.num_class)

    def forward(self, x):
        # print('input:', x.size())
        out = self.embedding(x)
        # print('out:', out.size())
        out = self.postion_embedding(out)
        # print('out:', out.size())
        for encoder in self.encoders:
            out = encoder(out)
        # print('out:', out.size())
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        # print('out:',out.size())
        out = self.fc1(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, ):
        super(Positional_Encoding, self).__init__()
        # self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None, dropout=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.transpose(-2, -1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        p_attn = F.softmax(attention, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, V), p_attn



class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, self.num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, self.num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, self.num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(self.num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        '''original code'''
        # Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        # K = K.view(batch_size * self.num_head, -1, self.dim_head)
        # V = V.view(batch_size * self.num_head, -1, self.dim_head)
        '''the follwing code was modified by LJW on 2021/3/4 17:30'''
        Q = Q.view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)##(batch_size, heads, sen_len, dim_head)
        K = K.view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_head, self.dim_head).transpose(1, 2)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        # context = self.attention(Q, K, V, scale, self.dropout)
        # context = context.view(batch_size, -1, self.dim_head * self.num_head)
        context, self.attn = self.attention(Q, K, V, scale, self.dropout)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.dim_head)

        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out




