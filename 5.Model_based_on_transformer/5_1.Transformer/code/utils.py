#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-02 19:11
# @Author  : 冯佳欣
# @File    : utils.py
# @Desc    : function 包

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy







# clones
def clones(module,N):
    '''
    product N identical layers
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# attention 方法
def attention(query,key,value,mask=None,dropout=None):
    '''
    计算 每个query的context 向量 和 attention_score,在transformer场景中，key和value的size一定相同
    :param query: [batch_size,q_len,d_k]
    :param key: [batch_size,k_len,d_k]
    :param value: [batch_size,k_len,d_k]
    :param mask: [batch_size,q_len,k_len]
    :param dropout:
    :return:
        context : [batch_size,q_len,d_k]
        attention_weight: [batch_size,q_len,k_len]
    '''
    d_k = query.size(-1)
    # [batch_size,q_len,k_len]
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill(mask==0,-1e9)
    p_attn = F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # p_attn [batch_size,q_len,k_len]
    # value [batch_size,k_len,d_k]
    # context_vec [batch_size,k_len,d_k]
    context_vec = torch.matmul(p_attn,value)
    return context_vec,p_attn

# masking
def padding_mask(seq_query,seq_key,pad_index=0):
    '''
    padding mask 用到三个地方 encoder,decoder, encoder-decoder
    :param seq_query: [batch,query_len]
    :param seq_key: [batch_size,key_len]
    :return:
        mask : [batch_size,query_len,key_len]
    '''
    query_len = seq_query.size(1)
    # pad is 0
    pad_mask = seq_key.eq(pad_index)
    # [batch_size,query_len,key_len]
    pad_mask = pad_mask.unsqueeze(1).expand(1,query_len,1)
    return pad_mask

def sequence_mask(seq):
    '''
    mask now and feature info
    产生一个上三角矩阵，上三角的值全为1，下三角的值全为0，对角线也是0
    :param seq: [batch_size,seq_len]
    :return:
        mask : [batch_size,seq_len,seq_len]
    '''
    batch_size,seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len,seq_len),dtype=torch.uint8),diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size,1,1)
    return mask

