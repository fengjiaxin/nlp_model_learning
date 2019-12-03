#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-02 11:14
# @Author  : 冯佳欣
# @File    : model.py
# @Desc    : model 及一些辅助工具包

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import clones,attention,padding_mask,sequence_mask





# MultiHead Attention方法
class MultiHeadedAttention(nn.Module):
    '''
    MultiHead(Q,K,V) = proj_linear(Concat(head_1,head_2,...,head_h))
    head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)
    '''
    def __init__(self,h=8,d_model=512,dropout_rate = 0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.query_linear = nn.Linear(d_model,d_model)
        self.key_linear = nn.Linear(d_model,d_model)
        self.value_linear = nn.Linear(d_model,d_model)
        self.proj_linear = nn.Linear(d_model,d_model)
        self.attn = None # 存储得分情况，为了方便之后的图显示
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,query,key,value,mask=None):
        '''
        其中key 和 value的 size 一定相同
        :param query: [batch_size,q_len]
        :param key: [batch_size,k_len]
        :param value: [batch_size,k_len]
        :param mask: [batch_size,q_len,k_len]
        :return:
            context_vec: [batch_size,q_len,k_len]
        '''
        batch_size = query.size(0)
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # split by heads
        # [batch_size * head,seq_len,d_k]
        query = query.view(batch_size * self.h,-1,self.d_k)
        key = key.view(batch_size * self.h,-1,self.d_k)
        value = value.view(batch_size * self.h,-1,self.d_k)

        if mask is not None:
            # [batch_size * h,q_len,k_len]
            mask = mask.expand(self.h,-1,-1)

        # context_vec [batch_size * h,q_len,d_k]
        context_vec,self.attn = attention(query,key,value,mask,self.dropout)
        context_vec = context_vec.contiguous().view(batch_size,-1,self.double() * self.h)
        return self.proj_linear(context_vec)

# layer normalize
class LayerNorm(nn.Module):
    def __init__(self,feature_size,eps=-1e6):
        super(LayerNorm,self).__init__()
        self.alpha = nn.Parameter(torch.ones(feature_size))
        self.beta = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.alpha * (x-mean)/(std + self.eps) + self.beta

# sublayer connection
class SublayerConnection(nn.Module):
    '''
    output = LayerNorm(x + dropout(sublarer(x)))
    '''
    def __init__(self,feature_size,dropout_rate = 0.1):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x,sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

# position-wise feed-forward network
class PositionwiseFeedForward(nn.Module):
    '''
    FFN(x) = relu(0,xW_1 + b_1)W_2 + b_2
    '''
    def __init__(self,d_model,d_ff,dropout_rate = 0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# embedding
class Embeddings(nn.Module):
    '''
    in the embedding layers, multiply those weights by d_model^(1/2)
    '''
    def __init__(self,d_model,vocab_size):
        super(Embeddings,self).__init__()
        self.word_emb = nn.Embedding(vocab_size,d_model)
        self.d_model = d_model

    def forward(self,x):
        return self.word_emb(x) * math.sqrt(self.d_model)

# positional encoding
class PositionalEncoding(nn.Module):
    '''
    偶数列：pe(pos,i) = sin(pos/10000^(i/d_model))
    奇数列：pe(pos,i) = cos(pos/10000^(i-1/d_model))
    '''
    def __init__(self,d_model,dropout_rate = 0.1,seq_len = 500):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # [seq_len,d_model]
        pe = np.array([
            [pos/np.power(10000,2.0*(j//2)/d_model) for j in range(d_model)] for pos in range(seq_len)
        ] )
        pe[:,0::2] = np.sin(pe[:,0::2])
        pe[:,1::2] = np.cos(pe[:,1::2])

        # [1,seq_len,d_model]
        pe = torch.from_numpy(pe).unsqueeze(0)

        # 在内存中定义一个常量，在模型保存和加载的时候可以写入和读出
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)




# encoder layer
class EncoderLayer(nn.Module):
    '''
    encoder layer has two sublayers
    the first is a multi-head self-attention
    the second is a position-wise fully connected feed-forward netword
    '''
    def __init__(self,d_model=512,h=8,d_ff=2018,dropout_rate=0.1):
        super(EncoderLayer,self).__init__()
        self.self_attn = MultiHeadedAttention(h,d_model,dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff,dropout_rate)
        self.sublayers = clones(SublayerConnection(d_model,dropout_rate),2)

    def forward(self,x,mask):
        x = self.sublayers[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayers[1](x,self.feed_forward)

class Encoder(nn.Module):
    '''
    core encoder is a stack of N layers
    '''
    def __init__(self,layer,N=6):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)

    def forward(self,x,mask):
        '''
        pass the input (and mask) through each layer in turn
        :param x:
        :param mask: pad mask
        :return:
        '''
        for layer in self.layers:
            x = layer(x,mask)
        return x

# decoder layer
class DecoderLayer(nn.Module):
    '''
    decoder is made of self-attn,src-attn,feed forward
    '''
    def __init__(self,d_model=512,h=8,d_ff=2048,dropout_rate=0.1):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadedAttention(h,d_model,dropout_rate=dropout_rate)
        self.src_attn = MultiHeadedAttention(h,d_model,dropout_rate=dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff,dropout_rate=dropout_rate)
        self.sublayers = clones(SublayerConnection(d_model,dropout_rate),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory
        x = self.sublayers[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x = self.sublayers[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayers[2](x,self.feed_forward)

class Decoder(nn.Module):
    '''
    generic N layer decoder with masking
    '''
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)

    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return x


# encoder-decoder architecture
class EncoderDecoder(nn.Module):
    '''
    a standard encoder-decoder architecture
    '''
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)

class Generator(nn.Module):
    '''
    define standard linear + softmax generation
    '''
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)

# Transformer
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size=100,
                 src_max_len=20,
                 tgt_vocab_size=100,
                 tgt_max_len=20,
                 num_layers=6,
                 d_model= 512,
                 h = 8,
                 d_ff =2048,
                 dropout_rate = 0.1):
        super(Transformer,self).__init__()

        encoder_layer = EncoderLayer(d_model,h,d_ff,dropout_rate=0.1)
        decoder_layer = DecoderLayer(d_model, h, d_ff, dropout_rate=0.1)
        src_pos = PositionalEncoding(d_model,dropout_rate,src_max_len)
        tgt_pos = PositionalEncoding(d_model,dropout_rate,tgt_max_len)

        # encoder module
        self.encoder = Encoder(encoder_layer,num_layers)
        # decoder module
        self.decoder = Decoder(decoder_layer,num_layers)
        # generator module
        self.generator = Generator(d_model,tgt_vocab_size)
        # src_emb
        self.src_embed = nn.Sequential(Embeddings(d_model,src_vocab_size),src_pos)
        # tgt emb
        self.tgt_embed = nn.Sequential(Embeddings(d_model,tgt_vocab_size),tgt_pos)


    def forward(self,src,tgt,src_mask,tgt_mask):
        '''
        :param src: [batch_size,src_vocab_size]
        :param tgt: [batch_size,tgt_vocab_size]
        :param src_mask: [batch_size,src_max_len,src_max_len]
        :param tgt_mask: 需要屏蔽pad 和 future 信息 [batch_size,tgt_max_len,tgt_max_len]
        :return:
            output : [batch_size,tgt_max_len,tgt_vocab_size]
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

