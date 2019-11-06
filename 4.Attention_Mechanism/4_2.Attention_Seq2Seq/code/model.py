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


class AttnDecoderGRU(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p=0.1,max_length=10):
        '''
        :param hidden_size:
        :param output_size: 输出的vocab 词汇表大小
        :param max_length: encoder 的序列长度
        '''
        super(AttnDecoderGRU,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 输出词语的词向量嵌入，嵌入后的维度是hidden_size
        self.embedding = nn.Embedding(output_size,hidden_size)
        # 计算该时刻的权重，输入是[embedded,hidden_state]
        self.attn = nn.Linear(self.hidden_size * 2,self.max_length)
        # 计算最后的output 输入是[embedded,context]
        self.attn_combine = nn.Linear(self.hidden_size * 2,self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)


    def forward(self,input,hidden,encoder_outputs):
        '''
        :param input: [1]
        :param hidden: [1,1,hidden_size]
        :param encoder_outputs: [max_length,hidden_size]
        :return:
            output:[1,output_size]
            hidden:[1,hidden_size]
            attn_weights:[1,max_length]
        '''
        # [1,1,hidden_size]
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        # [1,2 * hidden_size]
        embedded_hidden_concat = torch.cat((embedded[0],hidden[0]),1)
        # [1,max_length]
        e_attn = self.attn(embedded_hidden_concat)
        # softmax [1,max_length]
        attn_weights = F.softmax(e_attn,dim=1)
        # 计算context [1,1,hidden_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        # [1,2 * hidden_size]
        output = torch.cat((embedded[0],attn_applied[0]),1)
        # [1,1,hidden_size]
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        # output [1,1,hidden_size] hidden [1,1,hidden_size]
        output,hidden = self.gru(output,hidden)
        # [1,output_size]
        output = F.log_softmax(self.out(output[0]),dim=1)

        return output,hidden,attn_weights

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size)



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


