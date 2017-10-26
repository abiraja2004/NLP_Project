#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/26/2017 1:52 PM
# @Author  : Mingfei Jia
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm

import nltk

def count_noun(file_name):
    txt_file = open(file_name,'r',encoding='utf-8')
    file_lines = txt_file.readlines()
    long_line = ''
    for line in file_lines:
        long_line+=line
    long_token = nltk.tokenize.word_tokenize(long_line)
    #print(long_token)
    tags = nltk.pos_tag(long_token)
    count = nltk.Counter([j for i, j in tags if j.startswith('NN')])
    #total_count = sum(j for i, j in tags if j.startswith('NN'))
    print(count)
    #print(tags)
    #print(total_count)

count_noun("../data/set1/a1.txt")
count_noun("../data/set1/a2.txt")
count_noun("../data/set1/a3.txt")
count_noun("../data/set1/a4.txt")
count_noun("../data/set1/a5.txt")
count_noun("../data/set1/a6.txt")
count_noun("../data/set1/a7.txt")
count_noun("../data/set1/a8.txt")
count_noun("../data/set1/a9.txt")
count_noun("../data/set1/a10.txt")