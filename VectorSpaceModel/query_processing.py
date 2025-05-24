import json
from preprocessing import case_folding, remove_stopwords, lemmatization
from math import sqrt


def preprocess_query(query, stopwords_path):
    query = case_folding(query)
    query = remove_stopwords(query, stopwords_path)
    query = lemmatization(query)
    return query


def compute_query_vector(preprocessed_query, idf_dict):
    query_tf = {}
    for word in preprocessed_query.split():
        query_tf[word] = query_tf.get(word, 0) + 1

    query_vector = {}
    for word, tf in query_tf.items():
        if word in idf_dict:  # Only include words that exist in the corpus
            query_vector[word] = tf * idf_dict[word]
    return query_vector


def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) & set(vec2))
    norm1 = sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = sqrt(sum(v ** 2 for v in vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def rank_documents(query_vector, doc_vectors, alpha):
    scores = {}
    for doc_id, doc_vector in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score >= alpha:
            scores[doc_id] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_query(query, stopwords_path, tfidf_path, alpha=0.001):
    with open(tfidf_path, 'r', encoding='utf-8') as f:
        tfidf_data = json.load(f)

    idf_dict = tfidf_data['idf']
    doc_vectors = tfidf_data['doc_vectors']

    preprocessed = preprocess_query(query, stopwords_path)
    query_vector = compute_query_vector(preprocessed, idf_dict)
    return rank_documents(query_vector, doc_vectors, alpha)

