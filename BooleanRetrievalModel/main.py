import subprocess
import sys
import os
import preprocessing
import indexing

# This will run in current working directory
cwd = os.getcwd()

# Paths for different files eg rawData,ProcessedData,Stopwords,Indexes
corpus = os.path.join(cwd, "Raw_Data")
stopwords = os.path.join(cwd, "Stopword-List.txt")
output_dir = os.path.join(cwd, "Preprocessed_Data")
index_output_dir = os.path.join(cwd, "Indexes")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(index_output_dir, exist_ok=True)


def install_dependencies(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def preprocess_text():
    # extracting original data from the raw data folder
    for filename in os.listdir(corpus):
        if filename.endswith(".txt"):
            file_path = os.path.join(corpus, filename)
            print(f"Processing: {filename}")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            # Applying preprocessing steps to original raw data
            text = preprocessing.case_folding(text)  # Converting to lowercase
            text = preprocessing.remove_stopwords(text, stopwords)  # Removing stopwords
            text = preprocessing.stemming(text)  # Stemming

            # saving to Preprocessed_Data folder
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == '__main__':
    install_dependencies(["PyQt5", "nltk"])

    # preprocessing the original raw data
    preprocess_text()

    # Building INdexes, both inverted and positional
    inverted_index, positional_index = indexing.build_indexes(output_dir)
    indexing.save_indexes(inverted_index, positional_index, index_output_dir)

    # Calling the gui
    from GUI import search_bar

    search_bar()
