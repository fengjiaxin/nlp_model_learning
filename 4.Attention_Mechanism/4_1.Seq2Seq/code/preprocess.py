#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-01 10:12
# @Author  : 冯佳欣
# @File    : preprocess.py
# @Desc    : 数据预处理部分代码
import os
import torch
import time
import math
import re
import string
from string import digits

SOS_token = 0
EOS_token = 1


# 处理数据的类
class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word_freq = {}
        self.index2word = {}
        self.index2word = {0:'SOS',1:'EOS'}
        self.n_words = 2

    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word_freq[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_freq[word] += 1

def remove_punctuation_digit_lower(line):
    line = line.strip()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # 去掉特殊字符
    sentence = regex.sub('', line)

    # 去掉数字
    remove_digits = str.maketrans('', '', digits)
    sentence = sentence.translate(remove_digits)

    # 小写
    sentence = sentence.lower()
    return sentence.strip()

# 从文件中读取encode 和 decode的语料
def readLangs(base_dir,lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    cache_file = os.path.join(base_dir,'%s-%s.txt'%(lang1,lang2))
    lines = open(cache_file).read().strip().split('\n')

    # Split every line into pairs
    pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
        eng_index = 1
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        eng_index = 0

    return input_lang, output_lang, pairs,eng_index

MAX_LENGTH = 10

eng_prefixes = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re"
)


def filterPair(p,eng_index):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[eng_index].startswith(eng_prefixes)


def filterPairs(pairs,eng_index):
    return [handle_pair(pair) for pair in pairs if filterPair(pair,eng_index)]

def handle_pair(pair):
    return (remove_punctuation_digit_lower(pair[0]),remove_punctuation_digit_lower(pair[1]))

def prepareData(base_dir,lang1, lang2, reverse=False):
    input_lang, output_lang, pairs,eng_index = readLangs(base_dir,lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs,eng_index)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs




# input_lang, output_lang, pairs = prepareData('../data/cache','eng', 'fra', True)

# 显示时间
def timeSince(since,percent):
    '''
    :param since: 开始记录的time时刻
    :param percent: 已完成的百分比
    :return:
    '''
    now = time.time()
    pass_time = now - since
    all_time = pass_time/percent
    remain_time = all_time - pass_time
    return '%s (- %s)' % (asMinutes(pass_time),asMinutes(remain_time))

def asMinutes(s):
    '''
    将时间s转换成minute 和 second的组合
    :param s:
    :return:
    '''
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds'%(m,s)

if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('../data/cache','eng', 'fra', False)
    print('打印前10行')
    print(pairs[:10])