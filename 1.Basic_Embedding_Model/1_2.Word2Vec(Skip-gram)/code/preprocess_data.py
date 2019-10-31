#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-31 14:35
# @Author  : 冯佳欣
# @File    : preprocess_data.py
# @Desc    : 处理数据

import re
import pandas as pd
import string
from string import digits
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from collections import Counter

def remove_punctuation_digit_lower(line):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # 去掉特殊字符
    sentence = regex.sub('', line)

    # 去掉数字
    remove_digits = str.maketrans('', '', digits)
    sentence = sentence.translate(remove_digits)

    # 小写
    sentence = sentence.lower()

    # 将多个空格变为一个空格
    sentence = re.sub(' +', ' ', sentence)
    return sentence

# 将raw文件处理后存储的中间文件中
def process_raw(read_file = '../data/origin_data/sample_train.csv',\
                write_file = '../data/process_data/content_process.txt'):
    if not os.path.exists(write_file):
        # 目标是取出数据中的content列，将其作为一句话，然后去掉标点符号，数字，都转换为小写，然后写到新的文件中
        # pandas读取数据
        data_df = pd.read_csv(read_file, header=None, names=['label', 'title', 'content'])
        content_array = data_df.content.values

        with open(write_file, 'w') as w:
            for sentence in content_array:
                sent = remove_punctuation_digit_lower(sentence)
                w.write(sent + '\n')

# 统计中间文件的词频，并构建词表等信息
def construct_vocab(sentence_file,max_vocab_size):
    '''
    :param sentence_file: 中间文件
    :param max_vocab_size: 词汇表数量
    :return:
        idx_to_word:id和word的对应关系
        word_to_ix:word和ix的对应关系
        word_freqs:array[max_vocab_size] 每个单词的采样权重
    '''
    word_counts = Counter()
    # 文中单词出现的所有次数
    all_word_num = 0
    with open(sentence_file) as f:
        for line in f:
            vec = line.strip().split()
            all_word_num += len(vec)
            word_counts.update(vec)

    # 构建词表，选取出现频率最大的max_vocab_size,-1是为<unk>留位置
    vocab = dict(word_counts.most_common(max_vocab_size-1))
    vocab['<UNK>'] = all_word_num - np.sum(list(vocab.values()))

    idx_to_word = {idx:word for idx,word in enumerate(vocab.keys())}
    word_to_ix = {word:idx for idx,word in enumerate(vocab.keys())}

    # 统计每个单词的词频
    word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
    # 每个单词的频率
    word_freqs = word_counts/np.sum(word_counts)

    # 论文中提到将频率 3/4幂运算后，然后归一化，对negitave sample 有提升
    word_freqs = word_freqs ** (3./4.)
    word_freqs = word_freqs/np.sum(word_freqs)

    return idx_to_word,word_to_ix,word_freqs


def get_context_target_data(sentence_file,word_to_ix,window_size):
    '''
    将sentgence 按照word_to_ix 和window_size，获取每个中心单词的id 以及中心单词周围的context id
    注意：对于一句话 w1 w2 w3 w4 w5
    对于w1来说，没有上文单词，为了方便处理，这里的操作是将w4,w5 当作w1 的上文
    同理，对于w5来说，w1,w2 当作w5的 下文
    :param sentence_file:
    :param word_to_ix:
    :param window_size:
    :return:
        target_id_list:存储中心的id的列表 [w1,w2]
        context_ids_list: 列表里面存储的是target上下文单词id的列表 [[w2,w3,w4,w5],[w1,w5,w3,w4]]
    '''
    target_ids_list = []
    context_ids_list = []
    unk_id = word_to_ix['<UNK>']
    with open(sentence_file) as f:
        for line in f:
            sent_vec = line.strip().split()
            # 句子的长度
            sent_len = len(sent_vec)
            for target_word_pos,target_word in enumerate(sent_vec):
                target_word_id = word_to_ix.get(target_word,unk_id)
                # context 单词针对 target 位置的偏移
                # 获取目标单词的context_id_list
                temp_context_ids = []
                for context_pos in range(-window_size,window_size + 1):
                    if context_pos == 0:continue
                    context_word = sent_vec[(target_word_pos + context_pos) % sent_len]
                    context_word_id = word_to_ix.get(context_word,unk_id)
                    temp_context_ids.append(context_word_id)
                target_ids_list.append(target_word_id)
                context_ids_list.append(temp_context_ids)

    return target_ids_list,context_ids_list




class MyDataset(Dataset):
    def __init__(self,sentence_file,word_to_ix,word_freqs,window_size,k):
        super(MyDataset,self).__init__()
        self.word_freqs = torch.Tensor(word_freqs)

        target_ids,context_ids_list = get_context_target_data(sentence_file,word_to_ix,window_size)
        # 将列表转换成tensor
        self.target_ids = torch.Tensor(target_ids)
        self.context_ids_list = torch.Tensor(context_ids_list)
        self.window_size = window_size
        self.k = k

    def __len__(self):
        return self.target_ids.size(0)

    def __getitem__(self, index):
        '''
        这个function 返回以下数据进行训练
            - 中心词
            - 这个单词附近的positive 单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word_id = self.target_ids[index]
        pos_indices = self.context_ids_list[index]

        # negative sample
        neg_indices = torch.multinomial(self.word_freqs,self.k * self.window_size * 2,True)

        return center_word_id,pos_indices,neg_indices

def get_data_loader(sentence_file,word_to_ix,word_freqs,window_size,k,batch_size):
    dataset = MyDataset(sentence_file,word_to_ix,word_freqs,window_size,k)
    data_loader = data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader
