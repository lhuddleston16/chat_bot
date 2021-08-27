import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()


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