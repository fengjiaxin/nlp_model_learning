#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-30 11:08
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : fasttext model

import torch
import torch.nn as nn

class fastText(nn.Module):
    def __init__(self,word_embeddings,hidden_size,embed_size,classes):
        '''
        :param word_embeddings: numpy array [vocab_size,embedding_dim]
        '''
        super(fastText,self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.classes = classes

        # embedding layer
        self.embeddings = nn.Embedding(len(word_embeddings),self.embed_size)
        self.embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.embeddings.weight.data.requires_grad = False

        # hidden layer 1
        self.fc1 = nn.Linear(self.embed_size,self.hidden_size)

        # output layer
        self.fc2 = nn.Linear(self.hidden_size,self.classes)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        '''
        前向传播
        :param x:[batch_size,seq_len]
        :return: [batch,class]
        '''
        embeded_sent = self.embeddings(x) #[batch,seq_len,emb_dim]
        # 平均
        embeded_sent = embeded_sent.mean(1) # [batch,emb_dim]
        h = self.fc1(embeded_sent)
        z = self.fc2(h) # [batch,classes]
        return self.softmax(z)