import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def remove_stopwords(text, stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8', errors='ignore') as f:
        stopwords = set(f.read().splitlines())

    words = word_tokenize(text)
    filtered_text = ' '.join(word for word in words if word.lower() not in stopwords)
    return filtered_text

def case_folding(text):
    return text.lower()

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'\W+', ' ', text)

    words = word_tokenize(text)
    lemmatized_text = ' '.join(lemmatizer.lemmatize(word) for word in words)
    return lemmatized_text