import nltk


# nltk.download()



from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import json
import pickle

# basics of NLTK
# x = "Hi, hello how r u. When are you coming to India"

# print(word_tokenize(x))
# print(sent_tokenize(x),"\n")


# from nltk.corpus import wordnet
# words = wordnet.synsets('love')

# print(words)
# print(">> definition of second set is,",words[1].definition(),"\n")


# word = wordnet.synsets('male')
# print(word,"\n")


# syn = []
# for i in wordnet.synsets('AI'):
#     for j in i.lemmas():
#         syn.append(j.name())
# print("synonym:",syn,"\n")
# syn2 = []
# for synonym in wordnet.synsets('beautiful'):
#     for l in synonym.lemmas():
#         if l.antonyms():
#             syn2.append(l.antonyms()[0].name())
# print("antonym:",syn2)


lemmatizer = WordNetLemmatizer()
# print("\n",lemmatizer.lemmatize("loving",pos='v'))
# print("\n",lemmatizer.lemmatize('firing',pos='n'))


import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
import random


words = []
classes = []
documents = []
ignore_words = ['?','!']
json_file = open('intent.json').read()
intents = json.loads(json_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenizing the word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # creatig the doc
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


classes = sorted(list(set(classes)))

print(len(documents),"documents")

print(len(classes),"classes",classes)

print(len(words),"lemmatized words",words)



pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))






