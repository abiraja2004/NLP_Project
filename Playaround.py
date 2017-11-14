# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:43:03 2017

@author: jingx
"""
import pandas as pd
import numpy as np
import csv

# 
import nltk,re,pprint
import sklearn
from nltk.corpus import conll2000
from nltk.chunk.util import conlltags2tree
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from gensim import corpora, models, similarities

from itertools import groupby

nltk.internals.config_java("C:/Program Files (x86)/Java/jre1.8.0_151/bin/java.exe")
eng_tagger = StanfordNERTagger(model_filename = 'C:\\Users\\jingx\\Dropbox\\MSCF Course\\NLP\\stanford-ner-2017-06-09\\classifiers\\english.all.3class.distsim.crf.ser.gz',\
                               path_to_jar = 'C:\\Users\\jingx\\Dropbox\\MSCF Course\\NLP\\stanford-ner-2017-06-09\\stanford-ner.jar')
#print(eng_tagger.tag('Rami Eid is studying at Stony Brook University in NY'.split()))
a = eng_tagger.tag('Rami Eid is studying at Stony Brook University in NY and loves Mike'.split())

#for tag, chunk in groupby(a, lambda x:x[1]):
#    if tag != "O":
#        print("%-12s"%tag, " ".join(w for w, t in chunk))
#b = eng_parser.parse("Rami Eid is studying at Stony Brook University in NY".split())



eng_parser = StanfordParser(r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser.jar",r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser-3.8.0-models.jar")
#print(list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split())))


eng_parser = StanfordDependencyParser(r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser.jar",r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser-3.8.0-models.jar")
res = list(eng_parser.parse("the quick brown fox jumps over the lazy dog".split()))
#for row in res[0].triples():
#    print(row)

trainfile = r'C:\Users\jingx\Dropbox\MSCF Course\NLP\NLP_Project\data\set1\a6.txt'
with open(trainfile,encoding='utf8') as fin:
    train = fin.readlines()

train = list(map(lambda x: x.strip('\n'), train))
train = list(map(lambda x: x.strip(' '), train))
train = ' '.join(train)


sent_tokenize_list = sent_tokenize(train)

NE = dict()
for i in range(200,240):#range(len(sent_tokenize_list)):
    sent = sent_tokenize_list[i]
    a = eng_tagger.tag(word_tokenize(sent))
    for tag, chunk in groupby(a, lambda x:x[1]):
        if tag != "O":
            thisne =" ".join(w for w, t in chunk)
            if NE.get(tag):
                if thisne in list(NE[tag].keys()):
                    NE[tag][thisne].append(i)
                else:
                    NE[tag][thisne] = [i]
            else:
                NE[tag] = {thisne:[i]}
        #print("%-12s"%tag, " ".join(w for w, t in chunk))

q = 'Who was Milan\'s chief executive in 2008'

ques_tokenize_list = word_tokenize(q)

a = eng_tagger.tag(ques_tokenize_list)
eng_parser = StanfordDependencyParser(r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser.jar",r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser-3.8.0-models.jar")
res = list(eng_parser.parse(q.split()))
#for row in res[0].triples():
#    print(row)

qNE = dict()
for tag, chunk in groupby(a, lambda x:x[1]):
    if tag != "O":
            thisne =" ".join(w for w, t in chunk)
            if NE.get(tag):
                if thisne in list(NE[tag].keys()):
                    #NE[tag][thisne]
                    if qNE.get(thisne):
                        qNE[thisne].append(NE[tag][thisne])
                    else:
                        qNE[thisne] = NE[tag][thisne]
                    #class UnigramChunker(nltk.ChunkParserI):
whereto = []
for k, val in qNE.items():
    whereto.extend(val)

whereto = set(whereto)

targets = [sent_tokenize_list[i] for i in whereto]
targetsw = [word_tokenize(s) for s in targets]
dictionary = corpora.Dictionary(targetsw)
corpus = [dictionary.doc2bow(text) for text in targetsw]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=5)
vec_bow = dictionary.doc2bow(word_tokenize(q))
vec_lsi = lsi[vec_bow]
index = similarities.MatrixSimilarity(lsi[corpus])

sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])

answer_sent = targets[sims[0][0]]
eng_parser = StanfordDependencyParser(r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser.jar",r"C:\Users\jingx\Dropbox\MSCF Course\NLP\stanford-parser-full-2017-06-09\stanford-parser-3.8.0-models.jar")
ans_res = list(eng_parser.parse(answer_sent.split()))
for row in ans_res[0].triples():
    print(row)
#    def __init__(self, train_sents):
#        #train_sents is a complete tree of POS-tagged words (chunk tree).
#        print (train_sents)
#        #tree2conlltags returns a list of 3-tuples containing (word, tag, IOB-tag). 
#        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
#        #Entries in train_data are like ('NNS', 'B-NP'), ('IN', 'O') etc 
#        #print train_data
#        self.tagger = nltk.UnigramTagger(train_data)
#
##The parse() method has to be implemented as it's part of the nltk.ChunkParserI interface
#    def parse(self, sentence):
#        pos_tags = [pos for (word,pos) in sentence]
#        #Tagging the PoS tags with IOB chunk tags
#        tagged_pos_tags = self.tagger.tag(pos_tags)
#        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
#         #print chunktags
#         #chunktags are 'B-NP', 'I-NP' or 'O'
#         #zip combines two lists, creating tuples taking elements from each list 
#        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)] 
#        return nltk.chunk.conlltags2tree(conlltags)
#
#test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
#train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
#unigram_chunker = UnigramChunker(train_sents)
#print(unigram_chunker.evaluate(test_sents))