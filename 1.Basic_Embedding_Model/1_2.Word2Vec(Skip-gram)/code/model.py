#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-31 14:11
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : skip_gram model文件

import torch
import torch.nn as nn
import torch.nn.functional as F

class Skip_Gram(nn.Module):
    def __init__(self,vocab_size,embed_size):
        '''
        :param vocab_size: 此表数量
        :param embed_size: 词向量维度
        '''
        super(Skip_Gram,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        init_range = 0.5 / self.embed_size
        # target词嵌入矩阵
        self.target_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.target_embed.weight.data.uniform_(-init_range,init_range)

        # context词嵌入矩阵
        self.context_embed = nn.Embedding(self.vocab_size,self.embed_size)
        self.context_embed.weight.data.uniform_(-init_range,init_range)

    def forward(self,input_labels,pos_labels,neg_labels):
        '''
        :param input_labels: 中心词 [batch_size]
        :param pos_labels: 中心词周围 context window 出现过的单词 [batch_size,window_size * 2]
        :param neg_labels: 中心词周围没有出现过的单词，从negative sampling [batch_size,window_size * 2 * K]
        :return:
        '''

        # [batch_size,embed_size]
        input_embedding = self.target_embed(input_labels)
        # [batch_size,window_size * 2,embed_size]
        pos_embedding = self.context_embed(pos_labels)
        # [batch_size,window_size * 2 * k,embed_size]
        neg_embedding = self.context_embed(neg_labels)

        log_pos = torch.bmm(pos_embedding,input_embedding.unsqueeze(2)).squeeze()
        log_neg = torch.bmm(neg_embedding,-input_embedding.unsqueeze(2)).squeeze()

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)

        loss = log_pos + log_neg
        return -loss.mean()

    def input_embeddings(self):
        return self.target_embed.weight.data.numpy()
