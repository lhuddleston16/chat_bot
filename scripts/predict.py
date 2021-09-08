# Imports
import pickle
import numpy as np
import json
import random
from scripts.text_to_numbers import bag_of_words


def predict_response_class(sentence, model, word_list, response_class):
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
        return_list.append({"intent": response_class[r[0]], "probability": str(r[1])})
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


def simons_response(
    sentence,
    model,
    intents=json.loads(open("data/intents.json").read()),
    word_list=pickle.load(open("artifacts/word_list.pkl", "rb")),
    response_class=pickle.load(open("artifacts/response_class.pkl", "rb")),
):
    """Returns an automated response given a sentence"""
    #ints = predict_response_class(sentence, model, word_list, response_class)
    #res = get_response(ints, intents)
    #return res
    return model