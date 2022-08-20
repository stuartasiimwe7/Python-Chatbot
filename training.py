import random
import json
import pickle  
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
#To run these tensoflow imports you gotta install the CUDA Toolkit incase you run into an error as below
#https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_516.94_windows.exe

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)
words = [lemmatizer.lemmatize(word) for words in words if word not in ignore_letters]
words = sorted(set(words))

#pickle.dump(words, open('words.pkl','wb'))
#pickle.dump(words,open('classes.pkl','wb'))
