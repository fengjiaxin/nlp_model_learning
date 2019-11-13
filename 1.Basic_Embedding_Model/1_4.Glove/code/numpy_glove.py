#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 11:20
# @Author  : 冯佳欣
# @File    : numpy_glove.py
# @Desc    : 利用numpy 实现glove训练的过程

import numpy as np
import scipy
import scipy.sparse as sparse
from collections import Counter
import random


# 根据提供的预料库，构建此表
def build_vocab(corpus):
    '''
    build a vocabulary with word frequencies for an entire corpus
    return a dic 'w -> (i,f)'
    '''
    print('building vocab from corpus')

    vocab = Counter()
    for line in corpus:
        tokens = line.split()
        vocab.update(tokens)

    print('Done building vocab from corpus')

    return {word:(i,freq) for i,(word,freq) in enumerate(vocab.items())}



# 构建共现矩阵
def build_cooccur(vocab,corpus,window_size=5,min_count=None):
    '''
    buil a word co-occurrence list for the given corpus

    this function is a tuple generator,where each element \
    (reapresenting a coocuurence parir) is of the form (i_main,i_context,cooccurrence)

    :param vocab:
    :param corpus:
    :param window_size:
    :param min_count:
    :return:
    '''
    vocab_size = len(vocab)
    id2word = dict((i,word) for word,(i,_) in vocab.items())

    # collect cooccurrences internally as a sparse matrix for passable
    # indexing speed;we will convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size,vocab_size),dtype=np.float64)

    for i,line in enumerate(corpus):
        if i % 1000 == 0:
            print('building cooccurrence matrix: on line %i',i)

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i,center_id in enumerate(token_ids):
            # collect all word ids in left window of center word
            context_ids = token_ids[max(0,center_i - window_size):center_i]
            contexts_len = len(context_ids)

            for left_i,left_id in enumerate(context_ids):
                # distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0/float(distance)

                # build co-occurrence matrix symmetrically
                cooccurrences[center_id,left_id] += increment
                cooccurrences[left_id,center_id] += increment

    # now yield our tuple sequence
    for i,(row,data) in enumerate(zip(cooccurrences.rows,cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx,j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue
            yield i,j,data[data_idx]

# train glove
def train_glove(vocab,cooccurrences,vector_size=50,iterations=5,x_max=10,alpha=0.75,learning_rate=0.01,**kwargs):
    vocab_size = len(vocab)

    # 建立词向量 上半vocab_size存储i_main词向量，下半vocab_size存储i_context的词向量
    W = ((np.random.rand(vocab_size * 2,vector_size) - 0.5)/float(vector_size + 1))

    biases = ((np.random.rand(vocab_size * 2) - 0.5)/float(vector_size + 1))

    # training is done via adaptive gradient descent(AdaGrad),to make this word we need to store
    # the sum of squares of all previous gradients.

    # initialize all squared gradient sums to 1 so that our inital adaptive learning rate is
    # simply the global learning rate

    gradient_squared = np.ones((vocab_size * 2, vector_size),dtype=np.float64)
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    data = []
    for i_main, i_context, cur_freq in cooccurrences:
        # i_main 代表 center_id
        # i_context 代表 context_id
        # cur_freq 代表 pair 出现的次数

        tup = (
            W[i_main],
            W[i_context + vocab_size],
            biases[i_main:i_main + 1],
            biases[i_context + vocab_size:i_context + vocab_size + 1],
            gradient_squared[i_main],
            gradient_squared[i_context + vocab_size],
            gradient_squared_biases[i_main:i_main + 1],
            gradient_squared_biases[i_context + vocab_size:i_context + vocab_size + 1],
            cur_freq
        )
        data.append(tup)

    global_cost_list = []
    # begin training by iteraltively calling the run_iter function
    for i in range(iterations):
        random.shuffle(data)
        # begin train once iter
        global_cost = 0
        for (v_main,v_context,b_main,b_context,gradsq_W_main,gradsq_W_context,gradsq_b_main,gradsq_b_context,freq) in data:

            # calculate weight function f(X_ij)
            weight = ((cur_freq)/x_max) ** alpha if cur_freq < x_max else 1

            # calculate cost_inner = w_i^Tw_j + b_i + b_j - log(X_{ij})
            cost_inner = (v_main.dot(v_context) + b_main[0] + b_context[0] - np.log(cur_freq))

            # compute cost
            cost = weight * cost_inner
            global_cost += cost

            # with the cost calculated,we now need to compute gradients
            # compute gradients for word vector terms
            grad_main = weight * cost_inner * v_context
            grad_context = weight * cost_inner * v_main

            # compute gradients for bias terms
            grad_bias_main = weight * cost_inner
            grad_bias_context = weight * cost_inner

            # now peerform adaptive updates
            v_main -= (learning_rate * grad_main/np.sqrt(gradsq_W_main))
            v_context -= (learning_rate * grad_context/np.sqrt(gradsq_W_context))

            b_main -= (learning_rate * grad_bias_main/np.sqrt(gradsq_b_main))
            b_context -= (learning_rate * grad_bias_context/np.sqrt(gradsq_b_context))

            # update squared gradient sums
            gradsq_W_main += np.square(grad_main)
            gradsq_W_context += np.square(grad_context)
            gradsq_b_main += grad_bias_main ** 2
            gradsq_b_context += grad_bias_context ** 2
        global_cost_list.append(global_cost)
    return global_cost_list

if __name__ == '__main__':

    data_path = '../data/mini_content_process.txt'

    corpus = []
    with open(data_path, 'r') as f:
        for line in f:
            corpus.append(line.strip())

    vocab = build_vocab(corpus)
    cooccurrences = build_cooccur(vocab,corpus)
    global_cost = train_glove(vocab,cooccurrences)
    print(global_cost)






