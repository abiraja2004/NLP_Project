from nltk.parse.stanford import StanfordParser
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.corpus import wordnet
import nltk
import os
import sys
from nltk.stem.wordnet import WordNetLemmatizer

NERTagger = StanfordNERTagger(model_filename = 'stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz',\
                               path_to_jar = 'stanford-ner-2017-06-09/stanford-ner.jar')

POSTagger = StanfordPOSTagger('stanford-postagger-full-2017-06-09/models/english-bidirectional-distsim.tagger',
                                  'stanford-postagger-full-2017-06-09/stanford-postagger.jar')

LOCATIONS = ["in","at","on","inside","opposite","beside","above","behind","down"]

def checkNPVP(sentence):
    #HOME_PATH = "E:/CMU/Natural Language Processing/StanfordNLP/stanford-parser-full-2017-06-09"
    # HOME_PATH = "/home/stanford-parser-full/stanford-parser-full-2017-06-09"
    #os.environ['STANFORD_PARSER'] = HOME_PATH
    #os.environ['STANFORD_MODELS'] = HOME_PATH
    ENG_Parser = StanfordParser('stanford-parser-full-2017-06-09/stanford-parser.jar','stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar')
    parse_list = list(ENG_Parser.raw_parse(sentence))
    # print(parse_list)
    # print(parse_list[0].pos())
    # print(len(parse_list[0]))
    result = ""
    for line in parse_list:
        # print(line)
        if line.label() == 'ROOT':
            for obj in line:
                # print(obj.label())
                if obj.label() == 'S':
                    for sub in obj:
                        # if sub == 'NP':
                        # print(obj.leaves())
                        # print(sub.label())
                        result += sub.label()
    # print(result)
    if result == "NPVP.":
        return parse_list[0]
    return None


def getSentence(file_name, num_sentence):
    if sys.version_info < (3, 0):
        txt_file = open(file_name, 'r').read().decode('utf-8', 'ignore')
    else:
        txt_file = open(file_name, 'r', encoding='utf-8').read()
    sent_tokenize_list = sent_tokenize(txt_file)
    index = 0
    pron_list = ['he', 'she', 'they']
    for line in sent_tokenize_list:
        if index == num_sentence:
            break
        words = line.split()
        if words[0].lower() in pron_list:
            parse_tree = checkNPVP(line)
            if parse_tree:
                index += 1
                # print(genWhoQuestion(parse_tree))
                # print("Ready for who questions: " + line)
        else:
            parse_tree = checkNPVP(line)
            if parse_tree:
                what_question = genWhatQuestion(parse_tree)
                yn_question = genYesNoQuestion(parse_tree)
                where_question = getWhereQuestion(line)
                # if what_question:
                #     print("Ready for what questions: " + line)
                #     index += 1
                #     print(what_question)
                # if yn_question:
                #     print("Ready for yes/no question: "+line)
                #     index += 1
                #     print(yn_question)
                if where_question:
                    print("Ready for where questions: " + line)
                    index += 1
                    print(where_question)


def getWhereQuestion(sentence):
    words = word_tokenize(sentence)
    tagged_sent = POSTagger.tag(words)
    print(tagged_sent)
    result = []
    flag = False
    TENSE = ["do","does","did"]
    for item in tagged_sent:
        word = item[0]
        tag = item[1]
        if tag in ["VB","VBG","VBP"]:
            if word == "are":
                verb = "are"
            else:
                verb = TENSE[0]
                result.append(word)
        elif tag == "VBZ":
            if word == "is":
                verb == "is"
            else:
                verb = TENSE[1]
                # result.append(WordNetLemmatizer().lemmatize(word, 'v'))
                result.append(word)
        elif tag in ["VBD","VBN"]:
            if word in ["was","were"]:
                verb == word
            else:
                verb = TENSE[2]
                # result.append(WordNetLemmatizer().lemmatize(word,'v'))
                result.append(word)
        elif tag == "IN" and word in LOCATIONS:
            flag = True
            break
        else:
            result.append(word)
    if flag:
        return "Where "+verb+" "+" ".join(result)+" ?"

print(getWhereQuestion("The film are on 15 May 2011 in competition at the 2011 Cannes Film Festival"))

# def getWhenQuestion(sentence):
#     words = word_tokenize(sentence)
#     tagged_sent = POSTagger.tag(words)
#     print(tagged_sent)
#     result = []
#     flag = False
#     TENSE = ["do","does","did"]
#     for item in tagged_sent:
#         word = item[0]
#         tag = item[1]
#         if tag in ["VB","VBG","VBP"]:
#             verb = TENSE[0]
#             result.append(word)
#         elif tag == "VBZ":
#             verb = TENSE[1]
#             #result.append(WordNetLemmatizer().lemmatize(word, 'v'))
#             result.append(word)
#         elif tag in ["VBD","VBN"]:
#             verb = TENSE[2]
#             #result.append(WordNetLemmatizer().lemmatize(word,'v'))
#             result.append(word)
#         elif tag == "IN" and word in LOCATIONS:
#             flag = True
#             break
#         else:
#             result.append(word)
#     if flag:
#         return "Where "+verb+" "+" ".join(result)+" ?"



# import re
# result = re.match(r"""(?ix)             # case-insensitive, verbose regex
#     \b                    # match a word boundary
#     (?:                   # match the following three times:
#      (?:                  # either
#       \d+                 # a number,
#       (?:\.|st|nd|rd|th)* # followed by a dot, st, nd, rd, or th (optional)
#       |                   # or a month name
#       (?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)
#      )
#      [\s./-]*             # followed by a date separator or whitespace (optional)
#     ){3}                  # do this three times
#     \b                    # and end at a word boundary.""", "The film are on 15 May 2011")
#
# print(result)


def genWhoQuestion(parse_tree):
    words = parse_tree.leaves()
    return 'Who ' + " ".join(words[1:len(words) - 1]) + '?'


def genWhatQuestion(parse_tree):
    be_verbs = ['is', 'are', 'was', 'were']
    prp = ['It', 'This', 'That', 'These', 'Those']
    words = parse_tree.leaves()
    tags = parse_tree.pos()
    print(tags)
    for tag_s in parse_tree:
        for tag_npvp in tag_s:
            # for words in NP part
            if tag_npvp.label() == 'NP':
                # length of these words is the index of the first word of VP part
                be_verb_index = len(tag_npvp.leaves())
            # if the first word in VP part is be verb
            if tag_npvp.label() == 'VP' and tag_npvp.leaves()[0] in be_verbs:
                # check the pos of the next word
                (next_word, next_pos) = tags[be_verb_index + 1]
                # check the pos of next next word if exists
                if be_verb_index + 2 < len(words):
                    (next_next_word, next_next_pos) = tags[be_verb_index + 2]
                be_verb = tag_npvp.leaves()[0]
                # if the next word is adj.
                if next_pos.startswith('JJ'):
                    return "What " + be_verb + ' ' + ' '.join(words[be_verb_index + 1:len(words) - 1]) + '?'
                # if next two words are adj. and adv.
                # remove the adv. in question
                elif next_pos.startswith('RB') and next_next_pos and next_next_pos.startswith('JJ'):
                    return "What " + be_verb + ' ' + ' '.join(words[be_verb_index + 2:len(words) - 1]) + '?'
                # else if the length of NP part is longer than 3
                # question on the second part.
                elif be_verb_index > 3:
                    return "What " + be_verb + ' ' + words[0].lower() + ' ' + " ".join(words[1:be_verb_index]) + "?"
    return None


def genYesNoQuestion(parse_tree):
    be_verbs = ['is', 'are', 'was', 'were']
    prp = ['It', 'This', 'That', 'These', 'Those']
    words = parse_tree.leaves()
    tags = parse_tree.pos()
    #print(parse_tree)
    for tag_s in parse_tree:
        for tag_npvp in tag_s:
            if tag_npvp.label() == 'NP':
                # length of these words is the index of the first word of VP part
                be_verb_index = len(tag_npvp.leaves())
            # if the first word in VP part is be verb
            if tag_npvp.label() == 'VP' and tag_npvp.leaves()[0] in be_verbs:
                # check the pos of the next word
                (next_word, next_pos) = tags[be_verb_index + 1]
                # check the pos of next next word if exists
                if be_verb_index + 2 < len(words):
                    (next_next_word, next_next_pos) = tags[be_verb_index + 2]
                be_verb = tag_npvp.leaves()[0]
                # if the next word is adj.
                if next_pos.startswith('JJ'):
                    # for syn in wordnet.synsets(next_word):
                    #     for syn_word in syn.lemmas():
                    #         if syn_word.name() != next_word:
                    #             print(syn_word.name())
                    #             break
                    return be_verb + ' ' +words[0].lower()+' '+' '.join(words[1:be_verb_index])+' '+' '.join(words[be_verb_index+1:len(words)-1])+'?'
                # if next two words are adj. and adv.
                # remove the adv. in question
                elif next_pos.startswith('RB') and next_next_pos and next_next_pos.startswith('JJ'):
                    return be_verb + ' ' + words[0].lower() + ' ' + ' '.join(words[1:be_verb_index]) + ' '.join(words[be_verb_index + 2:len(words) - 1]) + '?'
    return None





#print(genWhatQuestion(checkNPVP("This is good.")))
#print(genYesNoQuestion(checkNPVP("He is amazingly available today.")))
#print(genYesNoQuestion(checkNPVP("He is good student.")))
#getSentence("./data/set4/a1.txt", 20)
