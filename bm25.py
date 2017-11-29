# bm25
# https://stackoverflow.com/questions/40966014/how-to-use-gensim-bm25-ranking-in-python
import math
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from six import iteritems



# from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/summarization/bm25.py

# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.3
EPSILON = 0.25   # the length of sentence does not matter much so


class BM25(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(document) / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in range(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights



stemmer = PorterStemmer()

def getbm25(question, file):
    bm25corpus = []
    rawcorpus = []
    for sentence in file:
        s = sent_tokenize(sentence)
        for ss in s:
            ssw = [stemmer.stem(w.lower()) for w in word_tokenize(ss)]
            bm25corpus.append(ssw)
            rawcorpus.append(ss)
    bm25obj = BM25(bm25corpus)
    average_idf = sum(map(lambda k: float(bm25obj.idf[k]), bm25obj.idf.keys())) / len(bm25obj.idf.keys())
    scores = bm25obj.get_scores(question, average_idf)
    return [(rawcorpus[i], scores[i]) for i in range(len(bm25corpus))]
