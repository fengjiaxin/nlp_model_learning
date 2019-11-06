#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-01 10:28
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : 不带有attention 机制的seq2seq 模型

import torch
import torch.nn as nn
import torch.nn.functional as F


# 训练思路：rnn 长度由自己设定，每个句子单独训练 step 和每个句子的长度相关

class EncoderGRU(nn.Module):
    def __init__(self,input_size,hidden_size):
        '''
        :param input_size: 在本模型中，就是input 的词汇表的大小
        :param hidden_size: the number of features in the hidden state h
        '''
        super(EncoderGRU,self).__init__()
        self.hidden_size = hidden_size
        # 需要将输入的特征维度嵌入到和hidden_size 一样的维度
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def initHidden(self):
        '''
        初始化hidden state
        h_0:[num_layers * num_directions,batch,hidden_size]
        '''
        return torch.zeros(1,1,self.hidden_size)

    def forward(self,input,hidden):
        '''
        :param input:[1]
        :param hidden:[1,1,hidden_size]
        :return:
        '''
        # [1,1,hidden_size]
        embedded = self.embedding(input).view(1,1,-1)
        input_embed = embedded

        # input_embed [seq_len,batch,input_size] -> [1,1,hidden_size]
        # hidden [num_layers * num_directions,batch,hidden_size] -> [1,1,hidden_size]
        # output [seq_len,batch,num_directions * hidden_size] ->[1,1,hidden_size]
        output,hidden = self.gru(input_embed,hidden)
        return output,hidden


class DecoderGRU(nn.Module):
    def __init__(self,hidden_size,output_size):
        '''
        :param hidden_size:
        :param output_size: 输出的vocab 词汇表大小
        '''
        super(DecoderGRU,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self,input,hidden):
        '''
        :param input: [1]
        :param hidden: [1,1,hidden_size]
        :return:
        '''

        # [1,1,hidden_size]
        input_embed = self.embedding(input).view(1,1,-1)
        input_embed = F.relu(input_embed)

        # input_embed: [seq_len,batch,hidden_size] -> [1,1,hidden_size]
        # hidden: [num_layers * num_directions,batch,hidden_size] -> [1,1,hidden_size]
        # output : [seq_len,batch,num_directions * hidden_size] -> [1,1,hidden_size]
        output,hidden = self.gru(input_embed,hidden)
        # self.out(output[0]) [1,output_size]
        output = self.softmax(self.out(output[0]))
        return output,hidden


