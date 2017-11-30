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

# NERTagger = StanfordNERTagger(
#     model_filename='stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz',
#     path_to_jar='stanford-ner-2017-06-09/stanford-ner.jar')
#
# POSTagger = StanfordPOSTagger('stanford-postagger-full-2017-06-09/models/english-bidirectional-distsim.tagger',
#                               'stanford-postagger-full-2017-06-09/stanford-postagger.jar')
#
# LOCATIONS = ["in", "at", "on", "inside", "opposite", "beside", "above", "behind", "down"]


def checkNPVP(parse_tree):
    #HOME_PATH = "E:/CMU/Natural Language Processing/StanfordNLP/stanford-parser-full-2017-06-09"
    # HOME_PATH = "/home/stanford-parser-full/stanford-parser-full-2017-06-09"
    #os.environ['STANFORD_PARSER'] = HOME_PATH
    #os.environ['STANFORD_MODELS'] = HOME_PATH
    # ENG_Parser = StanfordParser('stanford-parser-full-2017-06-09/stanford-parser.jar','stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar')
    #ENG_Parser = StanfordParser(model_path=HOME_PATH + "/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    #parse_list = list(ENG_Parser.raw_parse(sentence))
    #print(parse_tree)
    result = ""
    for line in parse_tree:
        # print(line)
        if line.label() == 'S':
            for obj in line:
                result += obj.label()
    # print(result)
    if result == "NPVP.":
        return parse_tree
    return None


def getSentence(file_name, num_sentence):
    HOME_PATH = "E:/CMU/Natural Language Processing/StanfordNLP/stanford-parser-full-2017-06-09"
    # HOME_PATH = "/home/stanford-parser-full/stanford-parser-full-2017-06-09"
    os.environ['STANFORD_PARSER'] = HOME_PATH
    os.environ['STANFORD_MODELS'] = HOME_PATH
    # ENG_Parser = StanfordParser('stanford-parser-full-2017-06-09/stanford-parser.jar','stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar')
    ENG_Parser = StanfordParser(model_path=HOME_PATH + "/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    # check version of Python in current environment
    if sys.version_info < (3, 0):
        txt_file = open(file_name, 'r').read().decode('utf-8', 'ignore')
    else:
        txt_file = open(file_name, 'r', encoding='utf-8').read()
    sent_tokenize_list = sent_tokenize(txt_file)
    print("Parsing the whole article, may take up to several minutes......")
    parsed_sentences = [sentence for line in ENG_Parser.raw_parse_sents(sent_tokenize_list) for sentence in line]
    index = 0
    for parse_tree in parsed_sentences:
        if index == num_sentence:
            break
        valid_parse_tree = checkNPVP(parse_tree)
        if valid_parse_tree:
            who_question = genWhoQuestion(valid_parse_tree)
            what_question = genWhatQuestion(valid_parse_tree)
            yn_question = genYesNoQuestion(valid_parse_tree)
            if who_question:
                index += 1
                print("Question " + str(index) + ": " + who_question)
            if what_question:
                index += 1
                print("Question " + str(index) + ": " + what_question)
            if yn_question:
                index += 1
                print("Question " + str(index) + ": " + yn_question)

    while index < num_sentence:
        index += 1
        print("Question " + str(index) + ": No more questions in this article...")


def getWhereQuestion(sentence):
    words = word_tokenize(sentence)
    tagged_sent = POSTagger.tag(words)
    print(tagged_sent)
    result = []
    flag = False
    TENSE = ["do", "does", "did"]
    for item in tagged_sent:
        word = item[0]
        tag = item[1]
        if tag in ["VB", "VBG", "VBP"]:
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
        elif tag in ["VBD", "VBN"]:
            if word in ["was", "were"]:
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
        return "Where " + verb + " " + " ".join(result) + " ?"


# print(getWhereQuestion("The film are on 15 May 2011 in competition at the 2011 Cannes Film Festival"))

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
    pron_list = ['he', 'she', 'they']
    if words[0].lower() in pron_list:
        return 'Who ' + " ".join(words[1:len(words) - 1]) + '?'
    return None

def genWhatQuestion(parse_tree):
    lemmatizer = WordNetLemmatizer()
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
            if tag_npvp.label() == 'VP' and be_verb_index < len(words)-2:
                (curr_word,curr_pos) = tags[be_verb_index]
                # check the pos of the next word
                (next_word, next_pos) = tags[be_verb_index + 1]
                # check the pos of next next word if exists
                (next_next_word, next_next_pos) = tags[be_verb_index + 2]
                (next_next_word,next_next_pos) = tags[be_verb_index+2]
                if curr_word not in be_verbs and curr_pos.startswith('VBD') and not next_pos.startswith('TO') and not next_pos.startswith('IN') and be_verb_index>3:
                    present_tense = lemmatizer.lemmatize(curr_word,'v')
                    return "What did "+' '+words[0].lower()+' '+' '.join(words[1:be_verb_index])+' '+present_tense+'?'
                # process be verbs
                elif tag_npvp.leaves()[0] in be_verbs:
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
                    elif be_verb_index > 1:
                        return "What " + be_verb + ' ' + words[0].lower() + ' ' + " ".join(words[1:be_verb_index]) + "?"

    return None


def genYesNoQuestion(parse_tree):
    be_verbs = ['is', 'are', 'was', 'were']
    prp = ['It', 'This', 'That', 'These', 'Those']
    lemmatizer = WordNetLemmatizer()
    words = parse_tree.leaves()
    tags = parse_tree.pos()
    # print(parse_tree)
    for tag_s in parse_tree:
        for tag_npvp in tag_s:
            if tag_npvp.label() == 'NP':
                # length of these words is the index of the first word of VP part
                be_verb_index = len(tag_npvp.leaves())
            # if the first word in VP part is be verb
            if tag_npvp.label() == 'VP' and be_verb_index > 1:
                if tag_npvp.leaves()[0] in be_verbs:
                    be_verb = tag_npvp.leaves()[0]
                    return be_verb.title() + ' ' + words[0].lower() + ' ' + ' '.join(words[1:be_verb_index]) + ' ' + ' '.join(words[be_verb_index + 1:len(words) - 1]) + '?'
                else:
                    (curr_word, curr_pos) = tags[be_verb_index]
                    if curr_pos.startswith('VBD'):
                        present_tense = lemmatizer.lemmatize(curr_word, 'v')
                        return "Did "+ words[0].lower() + ' ' + ' '.join(words[1:be_verb_index]) + ' ' + present_tense +' '+ ' '.join(words[be_verb_index + 1:len(words) - 1])+'?'

    return None

# print(genWhatQuestion(checkNPVP("This is good.")))
# print(genYesNoQuestion(checkNPVP("He is amazingly available today.")))
# print(genYesNoQuestion(checkNPVP("He is good student.")))
getSentence("./data/set3/a6.txt", 100)



