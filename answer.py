#!/usr/bin/env python3
import nltk,math,string,sys
from bm25 import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem.porter import *
from nltk.corpus import wordnet
from nltk import pos_tag
from itertools import groupby

stemmer = PorterStemmer()

HOME_PATH = "/home/stanford-parser-full/stanford-parser-full-2017-06-09"
NERTagger = StanfordNERTagger(model_filename = 'stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz',\
                               path_to_jar = 'stanford-ner-2017-06-09/stanford-ner.jar')

depParse =StanfordDependencyParser(HOME_PATH + "/stanford-parser.jar",HOME_PATH+'/stanford-parser-3.8.0-models.jar')

# NERTagger = StanfordNERTagger(model_filename = '/Users/jhl/Desktop/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz',\
#                                path_to_jar = '/Users/jhl/Desktop/stanford-ner-2017-06-09/stanford-ner.jar')

# depParse =StanfordDependencyParser("/Users/jhl/Desktop/stanford-parser-full-2017-06-09/stanford-parser.jar",
#             '/Users/jhl/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar')



YESNOSTART = set(["is",'are','am','was','were',
                'do','does','did','has','have','had','can',
                'could','would','might'])
NEGATION = set(['not','no',"n't","unlike"])

QUESTIONNE = {
        "who": "PERSON",
        "where": "LOCATION",
}

REASON = set(["because", "for","as","since", "considering","inasmuch","owing",
    "seeing"])

ARTICLETYPES = {
        'constellation': ["constellations", "constallation","zodiac"],
        'football':["soccer",'footballer','football'],
        'language':["language",'speaker'],
        'movie':['movie','film']
}

PRONOUNS = {'He','he','She','she','it','It','They','they','You','you'}
PEOPLEPRO =  {'He','he','She','she','They','they','You','you'}
numbersNLTKDontknow = {'eleven','twelve','thirteen','fourteen','fifteen','sixteen',
'seventeen','eighteen','nineteen'}
def similarity(s1,s2,idfDict):
    # helper function for tf-idf to calculate the cos similiarity
    a = set(s1)
    for t in s2:
        a.add(t)
    s1d = dict()
    s2d = dict()
    for w in s1:
        if w in s1d:
            s1d[w] += 1
        else:
            s1d[w] = 1
    for w in s2:
        if w in s2d:
            s2d[w] += 1
        else:
            s2d[w] = 1
    for w in a:
        if w not in s1d:
            s1d[w] = 0
        if w not in s2d:
            s2d[w] = 0

    s1tf = dict()
    s2tf = dict()
    for w in s1d:
        s1tf[w] = s1d[w]/len(s1)
    for w in s2d:
        s2tf[w] = s2d[w]/len(s2)

    s1idf = dict()
    s2idf = dict()

    for w in a:
        s1idf[w] = idfDict.get(w.lower(),0)
        s2idf[w] = idfDict.get(w.lower(),0)

    for w in a:
        s1d[w] = s1tf[w] * s1idf[w]
        s2d[w] = s2tf[w] * s2idf[w]
    # print(s1d,s2d)
    dotProd = 0
    norm1 = 0
    norm2 = 0
    for key in s1d:
        dotProd += s1d[key]*s2d[key]
        norm1 +=  s1d[key]**2
        norm2 +=  s2d[key]**2
    if norm1*norm2 == 0:
        return 0
    return dotProd/(math.sqrt(norm1)*math.sqrt(norm2))

def getIdfs(file):
    # a helper function to create an dictionary with idf values
    idfs = dict()
    sentenceNumber = 0
    for line in file:
        for sentence in sent_tokenize(line):
            ws = [stemmer.stem(w.lower()) for w in word_tokenize(sentence)]
            sentenceNumber += 1
            for wlower in set(ws):
                if wlower in idfs:
                    idfs[wlower] += 1
                else:
                    idfs[wlower] = 1
    for w in idfs:
        idfs[w] = 1 + math.log(sentenceNumber / idfs[w])
    return idfs

def best5(file,question):
    bestSim = 0
    bestAns = None
    docLength = 0
    idfDict = getIdfs(file)
    sims = []
    question = [stemmer.stem(w) for w in question[1:]]

    # lambdas to be tuned later
    lambda1 = 0.5 # lambda for the tf-idf written above
    lambda2 = 0.03 # lambda for the bm25 score from gensim bm25
    bm25res = getbm25(question, file)
    for sentence in bm25res:
        s = sentence[0]
        bm25sim = sentence[1]
        ssw = [stemmer.stem(w.lower()) for w in word_tokenize(s)]
        thisSim = similarity(question,ssw,idfDict)
        # print(thisSim, bm25sim)
        sims.append((s, (lambda1 * thisSim + lambda2 * bm25sim)))
    sortedSim = sorted(sims, key=lambda sentence: sentence[1],reverse = True)
    filtered = list(filter(lambda sentence: sentence [1] > 1.5, sortedSim))
    if filtered == []:
        return sortedSim[0:5]
    else:
        return filtered[0:5]

def questionType(q):
    # function that takes in a list of tokens to get the question type
    if q[0].lower() in YESNOSTART:
        return "yesno"
    return q[0].lower()

def rawToYesNo(qtokens, answer):
    # count the negation tokens and compare the number
    qneg = 0
    aneg = 0
    ansTokens = word_tokenize(answer)
    for tok in qtokens:
        if stemmer.stem(tok.lower()) in NEGATION:
            qneg += 1
    for tok in ansTokens:
        if stemmer.stem(tok.lower()) in NEGATION:
            aneg += 1
    antCount = 0
    for tok in qtokens:
        antonyms = []
        for syn in wordnet.synsets(tok):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms.append(stemmer.stem(l.antonyms()[0].name()))
        for tok in ansTokens:
            if stemmer.stem(tok) in antonyms:
                antCount += 1
    if qneg == aneg:
        if antCount % 2 == 1:
            return "No."
        return "Yes."
    else:
        if abs(qneg - aneg) == antCount:
            return "Yes."
        return "No."

# functions that should extract the information and produce answer.
def whatToAns(qtokens,rawAns):
    # what needs more info
    return rawAns

def whyToAns(qtokens,rawAns):
    return rawAns

def whenToAns(qtokens,rawAns):
    rawToks = word_tokenize(rawAns)
    # try to match the correct date using a algorithm like in who?
    tagged = NERTagger.tag(rawToks)
    merged = merge(tagged)
    try:
        return merged['DATE'][0] + "."
    except:
        return rawAns

def dist(qtokens,person,rawAns):
    ansTokL = [tok.lower() for tok in word_tokenize(rawAns)]
    if person.lower() in ansTokL:
        personI = ansTokL.index(person.lower())
    l = []
    for i in range(len(qtokens)):
        tok = qtokens[i]
        if tok in ansTokL:
            findRes = ansTokL.index(tok.lower())
            l.append(findRes)
    res = 0
    for n in l:
        res += abs(personI - n)
    return res

def whoToAns(qtokens,rawAns):
    rawToks = word_tokenize(rawAns)
    # well  I made this up :( maybe use dependency parsing?

    tagged = NERTagger.tag(rawToks)
    merged = merge(tagged)
    people = []
    for (w,t) in tagged:
        if t == "PERSON":
            people.append(w)
    if len(people) == 0:
        for pronoun in ['he','she','that','this']:
            if pronoun in rawAns.lower():
                return title
        return rawAns
    else:
        # find the closest PERSON to the tokens in questions
        bestDist = None
        for person in people:
            d = dist(qtokens,person,rawAns)
            if bestDist == None or d < bestDist:
                bestDist = d 
                bestPerson = person
        for p in merged['PERSON']:
            if bestPerson in p:
                return p + '.'
        return bestPerson + "."

def howToAns(qtokens,rawAns): 
    if qtokens[1] == "many":   
        parsed = [list(parse.triples()) for parse in depParse.raw_parse(rawAns)]
        nums = []
        for dep in parsed:
            for (thing, rel, what) in dep:
                if rel == 'nummod':
                    nums.append(what[0])   
        # pos tag raw Ans get cd
        for (word, tag) in (pos_tag(word_tokenize(rawAns))):
            if tag == "CD":
                nums.append(word)
        if nums != []:
            bestNum = None
            bestDist = None
            for num in nums:
                d = dist(qtokens,num,rawAns)
                if bestDist == None or d < bestDist:
                    bestDist = d 
                    bestNum = num
            if bestNum.isalpha():
                if bestNum[0].isalpha():
                    return bestNum[0].upper() + bestNum[1:] + "."
                return bestNum + '.'
    return rawAns
def whereToAns(qtokens,rawAns):
    return rawAns
def whichToAns(qtokens,rawAns):
    return rawAns  
def otherToAns(qtokens,rawAns):
    return rawAns


#https://stackoverflow.com/questions/30664677/extract-list-of-persons-and-organizations-using-stanford-ner-tagger-in-nltk 
def merge(netagged_words):
    NE = dict()
    for tag, chunk in groupby(netagged_words, lambda x:x[1]):
        if tag != "O":
            if tag not in NE :
                NE[tag] = []
            NE[tag].append( " ".join(w for w, t in chunk))
    return NE

def locateUsingNer(best5Sen, qtype, q):
    senLiteral = [sen[0] for sen in best5Sen]
    if qtype in ["yesno","what","which"]:
        return senLiteral[0]
    if qtype == "who" or qtype == "where" or qtype == "when": 
        for sen in senLiteral:
            tagged = NERTagger.tag(word_tokenize(sen))
            for (word,tag) in tagged:
                if qtype == "who" and (tag == "PERSON" or word in PEOPLEPRO):
                    return sen
                if qtype == "where" and tag == "LOCATION":
                    return sen
                if qtype == "when" and tag == "DATE":
                    return sen
                if qtype == "when" and (re.match(r'.*([1-3][0-9]{3})', sen))!=None:
                    return sen

    if qtype == "why":
        for sen in senLiteral:
            for w in word_tokenize(sen):
                if w.lower() in REASON:
                    return sen
    
    if qtype == "how":
        # print(best5Sen)
        qt = word_tokenize(q)
        if qt[1] == "many":
            for sen in senLiteral:
                parsed = [list(parse.triples()) for parse in depParse.raw_parse(sen)]
                for dep in parsed:
                    for (thing, rel, what) in dep:
                        if rel == 'nummod':
                            return sen
                for (word, tag) in (pos_tag(word_tokenize(sen))):
                    if tag == "CD":
                        return sen
                for num in numbersNLTKDontknow:
                    if num in sen.lower():
                        return sen
                for w in word_tokenize(sen):
                    if w.isdigit():
                        return sen
   
        # pos tag raw Ans get cd

    return senLiteral[0]

def getCategory(ff):
    firstParagraphs = ff[3]
    res = "other"
    for arttype in ARTICLETYPES:
        for keywords in ARTICLETYPES[arttype]:
            if keywords in firstParagraphs.lower():
                res = arttype
    if res == "other":
        for arttype in ARTICLETYPES:
            for keywords in ARTICLETYPES[arttype]:
                if keywords in ff[4].lower():
                    res = arttype
    return res

def testGet():
    for setNo in range(4):
        for articleNo in range(10):
            path = 'data/set%d/a%d.txt'%(setNo+1,articleNo+1)
            f = open(path)
            file = f.read().splitlines()

def postProcess(ans):
    ans = word_tokenize(ans)
    for pronoun in PRONOUNS:
        if pronoun in ans:
            index = ans.index(pronoun)
            ans[index] = title
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in ans]).strip()


def answerOne(file, question):
    global title, metadata

    f = open(file)
    ff = f.read().splitlines()                  # splitlines to avoid whitespace issue
    title = ff[0]
    category = getCategory(ff)
    qtokens = word_tokenize(question)           # tokenized question
    qtype = questionType(qtokens)
    best5Sen = (best5(ff, qtokens))   # string of the most similar sentence
    # print(best5Sen)
    if best5Sen[0][1] > 2.5:        # instead of directly returning, maybe make NER a parameter
        rawAnswer = best5Sen[0][0]    # string of the most similar sentence
    else:
        rawAnswer = locateUsingNer(best5Sen, qtype, question)
    # process the raw answer according to the type

    if questionType(qtokens) == "yesno":
        return rawToYesNo(qtokens,rawAnswer)

    finalAns = None
    if len(word_tokenize(rawAnswer)) - len(qtokens) < 10:
        return(postProcess(rawAnswer))
    if questionType(qtokens) == "what":
        return(postProcess( whatToAns(qtokens,rawAnswer)))
    if questionType(qtokens) == "when":
        return(postProcess( whenToAns(qtokens,rawAnswer)))
    if questionType(qtokens) == "why":
        return(postProcess( whyToAns(qtokens,rawAnswer)))
    if questionType(qtokens) == "how":
        return(postProcess(howToAns(qtokens,rawAnswer)))
    if questionType(qtokens) == "who":
        return(postProcess(whoToAns(qtokens,rawAnswer)))  
    if questionType(qtokens) == "where":
        return(postProcess(whereToAns(qtokens,rawAnswer)))
    if questionType(qtokens) == "which":
        return(postProcess(whichToAns(qtokens,rawAnswer)))
    else:
        return(postProcess(otherToAns(qtokens,rawAnswer)))


def answer(file, questions):
    q = open(questions)
    qs = q.read().splitlines()
    for question in qs:
        print(answerOne(file,question))

answer(sys.argv[1],sys.argv[2])

# (answer('data/set1/a6.txt','ourqs/David_Beckham.txt'))
# (answer('data/set3/a10.txt','ourqs/Esperanto.txt'))
# (answer('data/set2/a8.txt','ourqs/Hercules_(constellation).txt'))
# (answer('data/set3/a9.txt','ourqs/Java_(programming_language).txt'))
# (answer('data/set1/a10.txt','ourqs/John_Terry.txt'))
# (answer('data/set3/a6.txt','ourqs/Latin.txt'))
# (answer('data/set3/a7.txt','ourqs/Python_(programming_language).txt'))
# (answer('data/set2/a10.txt','ourqs/Crux.txt'))


# problematic questions:
'''
1. How many music albums in Esperanto in the nineties?
    sixties, seventies, eighties, nineties. cannot find the correct one. 

4. where was John Terry born?
    metadata
5. Which team does John play for?
6. What is CPython?
    too short
7. Is 10h 55.13m a right ascension coordinate?
    ......

'''


