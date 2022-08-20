import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
#import tflearn
#import tensorflow
import random
import json
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []
ignore_letters = ['?','!','.',',']

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs_x.append(word_list)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_letters ]
words = sorted(list(set(words)))

labels = sorted(labels)
#since neural networks only understand numbers - we are going to create a 
#bag representing all the words in any given pattern to be used to train the model - "One hot encoding"

training = []
output = []
out_empty = [0 for _ in range(len(labels))] 

for x, doc in enumerate(docs_x):
    bag = []
    word_list = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in word_list:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row [labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

    #training = numpy.array(training)
    output = np.array(output)