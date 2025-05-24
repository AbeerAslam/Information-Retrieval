import os
import math
import json
from collections import defaultdict
from nltk.tokenize import word_tokenize


def build_vsm(preprocessed_dir, output_dir):
    doc_term_freq = {}  # term frequencies for each doc
    document_frequency = defaultdict(int)  # number of docs each term appears in
    total_docs = 0

    print("Building TF and DF...")

    # Step 1: Read preprocessed documents and calculate TF and DF
    for filename in os.listdir(preprocessed_dir):
        if filename.endswith(".txt"):
            total_docs += 1
            file_path = os.path.join(preprocessed_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                words = word_tokenize(text)

            tf = defaultdict(int)
            seen_terms = set()

            for word in words:
                tf[word] += 1
                seen_terms.add(word)

            doc_term_freq[filename] = tf

            for term in seen_terms:
                document_frequency[term] += 1

    # Step 2: Compute TF-IDF
    doc_vectors = {}

    print("Computing TF-IDF weights...")
    for doc, term_freqs in doc_term_freq.items():
        tfidf_vector = {}
        for term, tf in term_freqs.items():
            df = document_frequency[term]
            idf = math.log(total_docs / df) if df else 0
            tfidf = tf * idf
            tfidf_vector[term] = tfidf
        doc_vectors[doc] = tfidf_vector

    # Step 3: Save TF-IDF vectors and IDF dictionary to a JSON file
    output_path = os.path.join(output_dir, "tfidf_vectors.json")

    idf_dict = {}
    for term, df in document_frequency.items():
        idf_dict[term] = math.log(total_docs / df) if df else 0

    output_data = {
        "doc_vectors": doc_vectors,
        "idf": idf_dict
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nTF-IDF vectors saved to {output_path}")

