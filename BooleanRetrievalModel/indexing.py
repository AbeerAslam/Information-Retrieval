import os
import json
from collections import defaultdict


def build_indexes(folder_path):
    # Word -> set of docs
    inverted_index = defaultdict(set)
    # Word -> {doc: [positions]}
    positional_index = defaultdict(lambda: defaultdict(list))

    # Sorting filenames numerically
    sorted_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split(".")[0]))

    for filename in sorted_files:
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            doc_id = filename.split(".")[0]

            with open(file_path, "r", encoding="utf-8") as f:
                words = f.read().split()

                for pos, word in enumerate(words):
                    inverted_index[word].add(doc_id)
                    positional_index[word][doc_id].append(pos)

    # Converting sets to sorted lists for JSON compatibility
    inverted_index = {word: sorted(list(docs), key=int) for word, docs in sorted(inverted_index.items())}

    return inverted_index, positional_index


def save_indexes(inverted_index, positional_index, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "inverted_index.json"), "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)

    with open(os.path.join(output_folder, "positional_index.json"), "w", encoding="utf-8") as f:
        json.dump(positional_index, f, indent=4)


def load_index(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)
