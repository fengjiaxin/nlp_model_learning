#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-11-01 10:12
# @Author  : 冯佳欣
# @File    : preprocess.py
# @Desc    : 去除原始文件中的一些不合法字符

import unicodedata
import re
import random

def generate_cache_data(read_file,write_file):
    with open(read_file,'r',encoding='utf-8') as f,open(write_file,'w') as w:
        print('begin handle data!')
        for line in f:
            pairs = line.strip().split('\t')
            encode_sent = normalizeString(pairs[0])
            decode_sent = normalizeString(pairs[1])
            w.write(encode_sent + '\t' + decode_sent + '\n')
        print('end handle data')



# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

if __name__ == '__main__':
    read_file = '../data/origin_data/eng-fra.txt'
    write_file = '../data/cache/eng-fra.txt'
    generate_cache_data(read_file,write_file)