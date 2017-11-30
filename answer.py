from nltk.corpus import wordnet
syns = wordnet.synsets("many")
print(wordnet.synsets("many"))
synonyms = []
antonyms = []

for syn in wordnet.synsets("significantly"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))