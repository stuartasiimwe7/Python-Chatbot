import random 
import json  
import nltk
import pickle
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

'''

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# remove duplicates
classes = sorted(list(set(classes)))
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
# output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


#MODEL
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Activation
from tensorflow.keras.models import Model
input_layer = Input(shape=(len(train_x[0])))
layer1 = Dense(100,activation='relu')(input_layer)
layer2 = Dense(50,activation='relu')(layer1)
output = Dense(len(train_y[0]),activation='sigmoid')(layer2)
#Creating a model
model = Model(inputs=input_layer,outputs=output)
model.summary()
model.compile(optimizer="Adam", loss="mse", metrics=['accuracy'])
model.fit(train_x, train_y, epochs=30, batch_size=1)

'''
