# imports
from simon_says import scrub_sentence
import nltk
nltk.download("punkt")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

# load data and initialize lists
word_list = []
reponse_class = []
documents = []
data_file = open("data/intents.json").read()
intents = json.loads(data_file)

intents["intents"][0]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        w = scrub_sentence(pattern)
        word_list.extend(w)
        # add documents in the corpus
        documents.append((w, intent["tag"]))
        # add to our reponse_class list
        if intent["tag"] not in reponse_class:
            reponse_class.append(intent["tag"])


# sort reponse_class
reponse_class = sorted(list(set(reponse_class)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# reponse_class = intents
print(len(reponse_class), "reponse_class", reponse_class)
# word_list = all words, vocabulary
print(len(word_list), "unique lemmatized word", word_list)
pickle.dump(word_list, open("artifacts/word_list.pkl", "wb"))
pickle.dump(reponse_class, open("artifacts/reponse_class.pkl", "wb"))

# Create bag of words data to use for training!
training = []
output_empty = [0] * len(reponse_class)

for doc in documents:
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # 1 if there is a match
    for w in word_list:
        bag.append(1) if w in pattern_words else bag.append(0)
    # Label the correct reponse_class with a 1
    output_row = list(output_empty)
    output_row[reponse_class.index(doc[1])] = 1
    training.append([bag, output_row])
# Randomize
random.shuffle(training)
training = np.array(training)
# Train and test
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Create model - 3 layers.
# First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents
model = Sequential()
model.add(Dense(100, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
# Using SGD with Nesterov
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
# fitting and saving the model
hist = model.fit(
    np.array(train_x), np.array(train_y), epochs=220, batch_size=5, verbose=1
)
model.save("artifacts/model.h5", hist)
print("model created")
