import nltk
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