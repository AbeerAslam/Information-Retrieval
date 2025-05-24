import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def remove_stopwords(text, stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8', errors='ignore') as f:
        stopwords = set(f.read().splitlines())

    words = word_tokenize(text)
    filtered_text = ' '.join(word for word in words if word.lower() not in stopwords)
    return filtered_text


def case_folding(text):
    return text.lower()


def stemming(text):
    stemmer = PorterStemmer()

    # ensuring - and / are replaced with spaces so porter stemmer works correctly on words like bag-of-words
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    words = word_tokenize(text)
    stemmed_text = ' '.join(stemmer.stem(word) for word in words)
    return stemmed_text
