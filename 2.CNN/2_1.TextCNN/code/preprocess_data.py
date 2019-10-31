#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-30 15:12
# @Author  : 冯佳欣
# @File    : preprocess_data.py
# @Desc    : 预处理数据

import pandas as pd
import re
import string
from string import digits
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self,data_file,w2ix_dict,max_len = 50):
        super(MyDataset,self).__init__()

        data_df = pd.read_csv(data_file, header=None, names=['label', 'title', 'content'])
        data_df['label'] = data_df['label'] -1
        self.labels = torch.from_numpy(data_df.label.values)
        self.content = data_df.content.values
        self.word_to_ix = w2ix_dict
        self.max_len = max_len

    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, index):
        label = self.labels[index]

        content = self.content[index]
        # 预处理文本
        sentence = remove_punctuation_digit_lower(content)
        document_encode = [self.word_to_ix.get(word,0) for word in sentence]
        if len(document_encode) < self.max_len:
            extended_sentences = [0] * (self.max_len - len(document_encode))
            document_encode.extend(extended_sentences)

        document_encode = torch.Tensor(document_encode[:self.max_len])
        return document_encode.long(),label


def remove_punctuation_digit_lower(line):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # 去掉特殊字符
    sentence = regex.sub('', line)

    # 去掉数字
    remove_digits = str.maketrans('', '', digits)
    sentence = sentence.translate(remove_digits)

    # 小写
    sentence = sentence.lower()
    return sentence

# def get_all_data(data_file,word_to_ix,seq_len):
#     '''
#     将数据转换成映射
#     :param data_file:
#     :param word_to_ix:
#     :param seq_len:
#     :return: tensor: train_data:[len(data_file),seq_len]
#              tensor: train_label:[len(data_file)]
#     '''
#     data_df = pd.read_csv(data_file,header=None,names=['label','title','content'])
#     # 将标签减一
#     data_df['label'] = data_df['label'] - 1
#     data_len = len(data_df)
#
#     train_sentence_data = [[0] * seq_len for i in range(data_len)]
#     train_label = []
#
#     for sentence_idx,row in data_df.iterrows():
#         train_label.append(int(row['label']))
#         sentence = remove_punctuation_digit_lower(row['content'])
#         for word_idx,word in enumerate(sentence):
#             if word_idx >= seq_len:break
#             if word in word_to_ix:
#                 mode_idx = word_to_ix[word]
#                 train_sentence_data[sentence_idx][word_idx] = mode_idx
#
#     return torch.from_numpy(np.array(train_sentence_data)),torch.from_numpy(np.array(train_label))


def get_data_loader(data_file,batch_size,word_to_ix,max_seq_len,mode='train'):
    training_params = {'batch_size':batch_size,
                       'shuffle':True}
    test_params = {'batch_size':batch_size,
                   'shuffle':False}

    if mode == 'train':
        dataset = MyDataset(data_file,word_to_ix,max_seq_len)
        training_loader = data.DataLoader(dataset, **training_params)
        return training_loader
    if mode == 'test':
        dataset = MyDataset(data_file,word_to_ix,max_seq_len)
        test_loader = data.DataLoader(dataset,*test_params)
        return test_loader
