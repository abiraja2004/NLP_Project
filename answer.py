import nltk,math,re
from bm25 import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger
from nltk.stem.porter import *

stemmer = PorterStemmer()

NERTagger = StanfordNERTagger(model_filename = '/Users/jhl/Desktop/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz',\
                               path_to_jar = '/Users/jhl/Desktop/stanford-ner-2017-06-09/stanford-ner.jar')

YESNOSTART = set(["is",'are','am','was','were',
                'do','does','did','has','have','had','can',
                'could','would','might'])
NEGATION = set(['not','no',"n't","unlike"])

QUESTIONNE = {
        "who": "PERSON",
        "where": "LOCATION",
}

TIME = set(["January", "February", "March", "April","May","June","July","August",
    "September","October","November","December"])
REASON = set(["because", "for","as","since", "considering","inasmuch","owing",
    "seeing"])


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
    question = [stemmer.stem(w) for w in question]

    # lambdas to be tuned later
    lambda1 = 0.7 # lambda for the tf-idf written above
    lambda2 = 0.3 # lambda for the bm25 score from gensim bm25
    bm25res = getbm25(question, file)

    for sentence in bm25res:
        s = sentence[0]
        bm25sim = sentence[1]
        ssw = [stemmer.stem(w.lower()) for w in word_tokenize(s)]
        thisSim = similarity(question,ssw,idfDict)
        sims.append((s, (lambda1*thisSim+lambda2*bm25sim)))
    sortedSim = sorted(sims, key=lambda sentence: sentence[1],reverse = True)
    return (sortedSim[0:5])

def questionType(q):
    # function that takes in a list of tokens to get the question type
    if q[0].lower() in YESNOSTART:
        return "yesno"
    return q[0].lower()

def rawToYesNo(qtokens, answer):
    # count the negation tokens and compare the number
    qneg = 0
    aneg = 0
    for tok in qtokens:
        if stemmer.stem(tok.lower()) in NEGATION:
            qneg += 1
    for tok in word_tokenize(answer):
        if stemmer.stem(tok.lower()) in NEGATION:
            aneg += 1
    if qneg == aneg:
        return "Yes."
    else:
        return "No."

# functions that should extract the information and produce answer.
def whatToAns(qtokens,rawAns):
    return rawAns
def whyToAns(qtokens,rawAns):
    return rawAns
def whenToAns(qtokens,rawAns):
    return rawAns
def whoToAns(qtokens,rawAns):
    return rawAns
def howToAns(qtokens,rawAns):
    return rawAns
def whereToAns(qtokens,rawAns):
    return rawAns
def whichToAns(qtokens,rawAns):
    return rawAns

def locateUsingNer(best5Sen, qtype, q):
    senLiteral = [sen[0] for sen in best5Sen]
    if qtype in ["yesno","what","which"]:
        return senLiteral[0]
    if qtype == "who" or qtype == "where": 
        for sen in senLiteral:
            tagged = NERTagger.tag(word_tokenize(sen))
            for (word,tag) in tagged:
                if qtype == "who" and tag == "PERSON":
                    return sen
                if qtype == "where" and tag == "LOCATION":
                    return sen
    if qtype == "when":
        for sen in senLiteral:
            for w in word_tokenize(sen):
                if w.isdigit() or w in TIME:
                    return sen
    if qtype == "why":
        for sen in senLiteral:
            for w in word_tokenize(sen):
                if w.lower() in REASON:
                    return sen
    
    if qtype == "how":
        qt = word_tokenize(q)
        if qt[1] == "many":
            for sen in senLiteral:
                for w in word_tokenize(sen):
                    if w.isdigit():
                        return sen
    return senLiteral[0]

def answer(file, question):
    f = open(file)
    ff = f.read().splitlines()                  # splitlines to avoid whitespace issue

    qtokens = word_tokenize(question)           # tokenized question

    qtype = questionType(qtokens)

    best5Sen = (best5(ff, qtokens))   # string of the most similar sentence
    # print(best5Sen)
    if best5Sen[0][1] > 0.8:
        rawAnswer = best5Sen[0][0]    # string of the most similar sentence
    else:
        rawAnswer = locateUsingNer(best5Sen, qtype, question)
    # process the raw answer according to the type
    if questionType(qtokens) == "yesno":
        return rawToYesNo(qtokens,rawAnswer)
    if questionType(qtokens) == "what":
        return whatToAns(qtokens,rawAnswer)
    if questionType(qtokens) == "when":
        return whenToAns(qtokens,rawAnswer)
    if questionType(qtokens) == "why":
        return whyToAns(qtokens,rawAnswer)
    if questionType(qtokens) == "how":
        return howToAns(qtokens,rawAnswer)
    if questionType(qtokens) == "who":
        return whoToAns(qtokens,rawAnswer)
    if questionType(qtokens) == "where":
        return whereToAns(qtokens,rawAnswer)
    if questionType(qtokens) == "which":
        return whichToAns(qtokens,rawAnswer)

# correctly answered questions:


print(answer('data/set2/a8.txt','Does Hercules have first or second magnitude stars?'))
print(answer('data/set2/a10.txt','Is Coalsack Nebula the most prominent dart nebula is the skies?'))
print(answer('data/set2/a8.txt','Is Hercules the fifth largest of the modern constellations?'))
print(answer('data/set1/a6.txt','Was Beckham the first England player ever to collect two red cards?'))
print(answer('data/set1/a9.txt','Does java combine the syntax for structured, generic and object-oriented programming?'))
print(answer('data/set3/a9.txt','Is the method name "main" a keyword in Java?'))
print(answer('data/set3/a9.txt','Does java use a garbage collector to manage memory?'))
print(answer('data/set3/a9.txt','Does java support C/C++ style pointer arithmetic?'))


# questions that can return the correct sentence but need to extract the answer.


print(answer('data/set1/a6.txt','Who was Milan\'s chief executive in 2008?'))
print(answer('data/set2/a8.txt','How far away is Mu Herculis from Earth?'))
print(answer('data/set2/a8.txt','What are the two bright globular clusters contained in Hercules?'))
print(answer('data/set3/a6.txt','Where is the Latin alphabet derived from?'))
print(answer('data/set3/a7.txt','How are conditional expressions written in Python?'))

print(answer('data/set3/a10.txt','why people can create new words of Esperanto?'))
print(answer('data/set3/a10.txt','Why Zamenhof created Esperanto?'))
print(answer('data/set3/a10.txt','How many music albums in Esperanto in the nineties?'))
print(answer('data/set3/a10.txt','What is the most widely spoken constructed language in the world?'))

print(answer('data/set3/a9.txt','What is WORA'))
print(answer('data/set3/a9.txt','Which version of java is supported for free by Oracle'))
print(answer('data/set3/a10.txt','How many countries used Esperanto?'))

# questions that cannot
# main problems: questions too short to match. NER?
print(answer('data/set1/a10.txt','where was John Terry born?'))
print("")
print("Q:what formed six diphthongs?")

print(answer('data/set3/a10.txt','what formed six diphthongs?'))
print("Q:when was Esperanto created?")

print(answer('data/set3/a10.txt','when was Esperanto created?'))

print("Q：How many native speakers speaker Esperanto?")


print(answer('data/set3/a10.txt','How many native speakers speaker Esperanto?'))
print("Q：what does Esperanto\'s name derive from?")

print(answer('data/set3/a10.txt','what does Esperanto\'s name derive from?'))

print(answer('data/set3/a9.txt','How many reported developers for java?'))
print("How many reported developers for java?")

print(answer('data/set2/a10.txt','What is Crus commonly known as?'))

# # incorrect "second"
# print(answer('data/set3/a9.txt','What is the second latest version of Java?'))

# # incorrect answer, few vs many
# print(answer('data/set3/a9.txt','Does java have as many low-level facilities as C and C++?'))





