#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-01 14:48
# @Author  : 冯佳欣
# @File    : train_test.py
# @Desc    : 训练模型

import torch
import random
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import EncoderGRU,DecoderGRU
from preprocess import prepareData,timeSince

teacher_forcing_rato = 0.5
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

base_dir = '../data/cache/'

# preparing training data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)



def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,critertion):
    '''
    :param input_tensor: [input_size,1]
    :param target_tensor: [target_size,1]
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param critertion:
    :param max_length:
    :return:
    '''
    # [1,1,hidden_size]
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)

    # 此时的encoder_hidden是encoder 后的全局向量

    decoder_input_init = torch.tensor([[SOS_token]])
    # [target_len + 1,1]
    decoder_inputs = torch.cat([decoder_input_init,target_tensor],dim=0)
    decoder_hidden = encoder_hidden


    use_teacher_forcing = True if random.random() < teacher_forcing_rato else False

    if use_teacher_forcing:
        # teacher forcing : feed the target as the next input
        for di in range(target_length):
            decoder_output,decoder_hidden = decoder(decoder_inputs[di],decoder_hidden)
            loss += critertion(decoder_output,target_tensor[di])
    else:
        decoder_input = decoder_input_init[0]
        for di in range(target_length):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
            topv,topi = decoder_output.topk(1)
            # 截断梯度流，梯度不从这里向前返回
            decoder_input = topi.squeeze().detach()
            loss += critertion(decoder_output,target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length

def trainIters(encoder,decoder,n_iters,print_every=10000,plot_every= 100,learning_rate = 0.01):
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimzier = optim.SGD(decoder.parameters(),lr=learning_rate)



    train_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1,n_iters + 1):
        train_pair = train_pairs[iter - 1]
        input_tensor = train_pair[0]
        target_tensor = train_pair[1]

        loss = train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimzier,criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

if __name__ == '__main__':
    hidden_size = 64
    input_lang, output_lang, pairs = prepareData(base_dir,'eng', 'fra', True)
    encoder = EncoderGRU(input_lang.n_words,hidden_size)
    decoder = DecoderGRU(hidden_size,output_lang.n_words)
    trainIters(encoder,decoder,10000,print_every=50)
    torch.save(encoder.state_dict(),'../model/encoder_param.pkl')
    torch.save(decoder.state_dict(), '../model/decoder_param.pkl')



