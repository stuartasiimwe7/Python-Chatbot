import random 
import json  
import nltk
import pickle
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

with open('intents.json') as json_data:
    intents = json.load(json_data)

lemmatizer = WordNetLemmatizer() 
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

# stem and lower each word
words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_letters]
words = sorted(set(words))
# remove duplicates
classes = sorted(list(set(classes)))

# Pickle module for serializing objects i.e converts python 
# objects like lists/dictionaries into byte streams (0s & 1s) 
with open('words.pkl', 'wb') as fh:
   pickle.dump(words, fh)

with open('classes.pkl', 'wb') as gh:
   pickle.dump(classes, gh)


print("Done")


#To be looked into later on
'''
training = []
output = []

# create an empty array for our output
output_empty = [0] * len(classes)

#training set
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [lemmatizer.lemmatize (word.lower()) for word in pattern_words]
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
#Hullo today is 16th of November
# create train and test lists
#But i really like this evening
train_x = list(training[:,0]))

train_y = list(training[:,1])

#The Model
import tensorflow as tf
from keras.layers import Dense,Input,Activation
from keras.models import Model
from keras.engine import base_layer 

#model = Sequential()
input_layer = Input(shape=(len(train_x[0])))
layer1 = Dense(100,activation='relu')(input_layer)
layer2 = Dense(50,activation='relu')(layer1)
output = Dense(len(train_y[0]),activation='sigmoid')(layer2)

#Creating a model
model = Model(inputs=input_layer,outputs=output)
model.summary()

model.compile(optimizer="Adam", loss="mse", metrics=['accuracy'])
model.fit(train_x, train_y, epochs=30, batch_size=1)

#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = sentence.split()
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

#creating a data structure to hold user context
context = {}
ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words).reshape((1,-1))])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

#check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        val = random.choice(i['responses'])
                        #speech = Speech(val, lang)
                        #sox_effects = ("speed", "1.0")
                        #speech.play(sox_effects)
                        # a random response from the intent
                        return print(val)
    results.pop(0)

'''

