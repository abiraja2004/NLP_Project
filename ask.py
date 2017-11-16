# encoding=utf8
from nltk.tokenize import sent_tokenize
from nltk.parse.stanford import StanfordParser
import nltk
import sys
import re

pronoun = ["she","he","it"]
pronouns = ["his","her","its"]

def generateCandidateSentence(file_name, num_sentence):
    file = open(file_name,'r').read().decode("utf8")
    sent_tokenize_list = sent_tokenize(file)
    #sent_tokenize_list =  [x.encode("utf8") for x in sent_tokenize_list]
    eng_parser = StanfordParser('stanford-parser-full-2017-06-09/stanford-parser.jar','stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar')
    num = 0
    name = sent_tokenize_list[0].split("\n")[0]
    #print(name)
    for sent in sent_tokenize_list:
        #filter some sentences:
        tmp_s = sent.split('\n')
        for s in tmp_s:
            if len(s) < 5:
                continue
            if checkNPVP(s, eng_parser):
                for p in pronoun:
                    if findWholeWord(p):
                       s = re.sub(p, name, s.lower(), count = 1);
                print(s)
                num += 0
            if num == num_sentence:
                break
    if num < num_sentence:
        for i in range(num_sentence - num):
            print("None")

def checkNPVP(sentence, eng_parser):
    parse_list = list(eng_parser.parse(sentence.split()))
    flag = False
    for a in parse_list[0]:
        if type(a) is nltk.tree.Tree and a.label() == 'S':
            tmp = ""
            for b in a:
                tmp += b.label()
            if tmp == "NPVP":
                flag = True
    return flag


def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

if __name__ == "__main__":
    file_name = sys.argv[1]
    num_sentence = int(sys.argv[2])
    generateCandidateSentence(file_name, num_sentence)