#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-30 19:37
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : textCNN model

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,vocab_array,label_num,filter_num,filter_sizes,vocab_size,embedding_dim,static,fine_tune,dropout):
        super(TextCNN,self).__init__()

        label_num = label_num  #标签的个数
        filter_num = filter_num # 每种卷积核的个数
        filter_sizes = [int(fsz) for fsz in filter_sizes.split(',')] # 卷积和的大小

        vocab_size = vocab_size
        embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        # 如果使用预训练词向量
        if static:
            self.embedding.weight.data.copy_(torch.from_numpy(vocab_array))
            if not fine_tune:
                self.embedding.weight.data.requires_grad = False


        # Conv2d input:(batch_size,c_in,H,W)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,filter_num,(fsz, embedding_dim)) for fsz in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num,label_num)

    def forward(self,x):
        '''
        :param x: [batch_size,seq_len]
        :return:
        '''
        # [batch_size,seq_len,embedding_dim]
        x = self.embedding(x)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 需要reshape [batch_size,1,seq_len,embedding_dim]
        x = x.view(batch_size,1,seq_len,self.embedding_dim)

        # 经过卷积运算，x中的每个运算结果维度为[batch_size,filter_num,h_out,1]
        x = [F.relu(conv(x)) for conv in self.convs]

        # 卷积池化 经过最大池化层，维度变为[batch_size,filter_num,h=1,w=1]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度(batch_size,filter_num,h,w) reshape (batch_size,filter_num * h * w)
        x = [x_item.view(x_item.size(0),-1) for x_item in x]

        # 将不同卷积和特征组合起来 [batch_size,sum: outchannel * w * h]
        x = torch.cat(x,1)

        x = self.dropout(x)

        # 全连接层
        logits = self.linear(x)
        return logits



