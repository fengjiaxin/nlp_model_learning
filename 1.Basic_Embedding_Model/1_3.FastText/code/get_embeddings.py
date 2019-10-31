#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-10-30 15:40
# @Author  : 冯佳欣
# @File    : get_embeddings.py
# @Desc    : 获取词向量
import pickle
import numpy as np
import os

def get_embedding(glove_file = '../data/word2vec_data/vocab_embedding.txt'):
    '''
    获取vocab_array,word2ix,ix2word
    :param glove_file:
    :return:
    '''


    vocab_array_pkl = '../data/pickle_data/vocab_embedding.pkl'
    word2ix_pkl = '../data/pickle_data/word_to_ix.pkl'
    ix2word_pkl = '../data/pickle_data/ix_to_word.pkl'
    embedding_dim = 50
    if not os.path.exists(vocab_array_pkl) and not os.path.exists(word2ix_pkl) and not os.path.exists(ix2word_pkl):
        vocab_list = ['<UNK>']
        vocab_matrix = [[0.0] * embedding_dim]

        with open(glove_file) as f:
            for line in f:
                vec = line.strip().split(' ')
                # print(vec)
                word = vec[0]
                vocab_list.append(word)
                word_vec = [float(x) for x in vec[1:]]
                # print(len(word_vec))
                assert len(word_vec) == embedding_dim
                vocab_matrix.append(word_vec)

        vocab_array = np.array(vocab_matrix)

        word_to_ix = {}
        ix_to_word = {}

        for ix, word in enumerate(vocab_list):
            word_to_ix[word] = ix
            ix_to_word[ix] = word


        with open(vocab_array_pkl, 'wb') as w:
            pickle.dump(vocab_array, w)

        with open(word2ix_pkl, 'wb') as w:
            pickle.dump(word_to_ix, w)

        with open(ix2word_pkl, 'wb') as w:
            pickle.dump(ix_to_word, w)
    # 载入
    vocab_array = pickle.load(open(vocab_array_pkl,'rb'))
    word_to_ix = pickle.load(open(word2ix_pkl,'rb'))
    ix_to_word = pickle.load(open(ix2word_pkl,'rb'))
    return vocab_array,word_to_ix,ix_to_word