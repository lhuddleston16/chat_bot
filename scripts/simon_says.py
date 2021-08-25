import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
import pickle
import numpy as np
from keras.models import load_model

model = load_model("artifacts/model.h5")
import json
import random

intents = json.loads(open("data/intents.json").read())
word_list = pickle.load(open("artifacts/word_list.pkl", "rb"))
reponse_class = pickle.load(open("artifacts/reponse_class.pkl", "rb"))
stop_words = set(stopwords.words('english'))

def scrub_sentence(sentence):
    """Break sentence into stem words and remove stop words."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    sentence_words = [ps.stem(w) for w in sentence_words]
    sentence_words = [w for w in sentence_words if not w.lower() in stop_words]
    sentence_words = sorted(list(set(sentence_words)))
    return sentence_words


def bag_of_words(sentence, word_list, show_details=True):
    """Using total words (word_list) create a bag of words"""
    # tokenize the pattern
    sentence_words = scrub_sentence(sentence)
    # bag of words
    bag = [0] * len(word_list)
    for s in sentence_words:
        for i, w in enumerate(word_list):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_reponse_class(sentence, model):
    """Input sentence and predict response class"""
    # Make prediction
    p = bag_of_words(sentence, word_list, show_details=False)
    res = model.predict(np.array([p]))[0]
    # Filter predictions below threshold of .25
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by highest probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": reponse_class[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    """Returns a automated response for a given response class"""
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def simons_response(text):
    """Returns an automated response given a sentence"""
    ints = predict_reponse_class(text, model)
    res = get_response(ints, intents)
    return res
