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

from get_embeddings import get_embedding
from preprocess_data import get_data_loader
from model import fastText

# 训练模型
def train_model(model,train_iter,epoches,lr):
    logging.info('## 1.begin training  ')
    # 将模式设置为训练模式
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoches):
        for batch_idx,(batch_x,batch_y) in enumerate(train_iter):
            batch_size = batch_x.size()[0]
            # 清除所有优化的梯度
            optimizer.zero_grad()
            # 传入数据并向前传播获取输出
            output = model(batch_x)
            loss = criterion(output,batch_y)
            loss.backward()
            optimizer.step()
            if batch_idx % 1000 == 0:
                logging.info("train epoch=" + str(epoch) + ",batch_id=" + \
                         str(batch_idx) + ",loss=" + str(loss.item() / batch_size))


# 测试模型
def model_test(model,test_iter):
    logging.info('begin test')
    # 将模式设置为eval
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i,(batch_x,batch_y) in enumerate(test_iter):
            logging.info('test batch_id = ' + str(i))
            outputs = model(batch_x)

            _,predicted = torch.max(outputs.data,1)
            total += batch_x.size()[0]
            correct += (predicted == batch_y).sum().item()
        logging.info('Accuracy of the network on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    glove_file = '../data/word2vec_data/vocab_embedding.txt'
    sample_train_file = '../data/origin_data/sample_train.csv'
    sample_test_file = '../data/origin_data/sample_test.csv'
    model_param_file = '../model/fasttext_model_param.pkl'

    # 超参数
    sentence_max_size = 50  # 每篇文章的最大词数量
    batch_size = 64
    epoches = 2
    emb_dim = 50 # 词向量维度
    lr = 0.0001
    hidden_size = 20
    label_size = 4

    # 获取词向量信息
    vocab_array,word_to_ix,ix_to_word = get_embedding(glove_file)
    train_iter = get_data_loader(sample_train_file,batch_size,word_to_ix,sentence_max_size)
    test_iter = get_data_loader(sample_test_file,batch_size,word_to_ix,sentence_max_size)

    # 定义模型
    model = fastText(vocab_array,hidden_size,emb_dim,label_size)

    # 训练
    logging.info('开始训练模型')
    train_model(model,train_iter,epoches,lr)

    # 模型保存
    torch.save(model.state_dict(),model_param_file)

    logging.info('开始测试模型')
    model_test(model,test_iter)
