#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-03 08:25
# @Author  : 冯佳欣
# @File    : data_utils.py
# @Desc    : 数据生成包

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
from utils import padding_mask,sequence_mask

# batches and masking
class Batch:
    '''
    object for holding a batch of data with mask during training
    '''
    def __init__(self,src,tgt=None,pad_index=0):
        '''
        :param src: [batch_size,max_src_len]
        :param tgt: [batch_size,max_tgt_len]
        :param pad:
        '''
        self.src = src
        self.src_mask = padding_mask(src,src,pad_index)
        if tgt is not None:
            self.tgt = tgt[:,:-1]
            self.tgt_y = tgt[:,1:]
            self.tgt_mask =

    @staticmethod
    def make_tgt_mask(tgt,pad_index=0):
        '''
        create a mask to hid padding and future words
        '''
        pad_mask =

