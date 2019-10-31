#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-30 15:49
# @Author  : 冯佳欣
# @File    : train_test.py
# @Desc    : 训练测试

import logging
logging.basicConfig(level=logging.INFO)
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess_data import get_data_loader,process_raw,construct_vocab
from model import Skip_Gram



if __name__ == '__main__':

    # 超参数
    k = 50  # number of negative samples
    window_size = 2
    batch_size = 64
    max_vocab_size = 10000
    epoches = 2
    emb_dim = 50 # 词向量维度
    lr = 0.001

    sample_train_file = '../data/origin_data/sample_train.csv'
    sentence_file = '../data/process_data/content_process.txt'
    model_param_file = '../model/skip_gram_param.pkl'
    word2vec_file = '../data/word2vec/vocab_embedding.txt'

    process_raw(sample_train_file,sentence_file)
    idx_to_word, word_to_ix, word_freqs = construct_vocab(sentence_file,max_vocab_size)


    # 定义模型
    model = Skip_Gram(max_vocab_size,emb_dim)
    model.load_state_dict(torch.load(model_param_file))

    torch.save(model.state_dict(),model_param_file)
    logging.info('保存模型参数')

    logging.info('保存word_embedding')
    word_embedding_array = model.input_embeddings()
    with open(word2vec_file,'w') as w:
        for i,word_vec in enumerate(word_embedding_array):
            word = idx_to_word[i]
            vec_str = ' '.join(['%.4f'%x for x in word_vec])
            w.write(word + ' ' + vec_str + '\n')
    logging.info('保存矩阵成功')
