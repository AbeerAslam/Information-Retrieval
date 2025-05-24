import subprocess
import sys
import os
import preprocessing
import VSM
from query_processing import search_query

# This will run in the current working directory
cwd = os.getcwd()

# Paths for different files eg rawData,ProcessedData,Stopwords,Indexes
corpus = os.path.join(cwd, "Raw_Data")
stopwords = os.path.join(cwd, "Stopword-List.txt")
preprocessed_dir = os.path.join(cwd, "Preprocessed_Data")
vector_dir = os.path.join(cwd, "Vectors")

os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)


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
            text = preprocessing.lemmatization(text)  # Lemmatization

            # saving to Preprocessed_Data folder
            output_path = os.path.join(preprocessed_dir, filename)
            with open(output_path, "w", encoding="utf-8", errors='ignore') as f:
                f.write(text)


def compare_results(got_result, expected_result):
    # Extract filenames from the got_result and strip the '.txt' extension
    got_filenames = {file.replace('.txt', '') for file, score in got_result}

    # Convert the expected_result (which has no .txt) to a set of strings
    expected_filenames = set(map(str, expected_result))  # Ensuring expected result is in string form

    # Compare the two sets of filenames
    if got_filenames == expected_filenames:
        return "match"
    else:
        return "no match"


if __name__ == '__main__':
    install_dependencies(["PyQt5","nltk"])
    # Check if the TF-IDF vectors file exists in the Vectors folder
    tfidf_file_path = os.path.join(vector_dir, "tfidf_vectors.json")

    if not os.path.exists(tfidf_file_path):
        # File doesn't exist, so preprocess and build vectors
        print("TF-IDF vectors not found. Proceeding with preprocessing and vector building...")

        # Preprocess the original raw data
        preprocess_text()

        # Build vectors for each document
        VSM.build_vsm(preprocessed_dir, vector_dir)
    else:
        # File exists, skip preprocessing and vector building
        print(f"TF-IDF vectors already exist at {tfidf_file_path}. Skipping preprocessing and vector building.")



# Calling the gui
from GUI import search_bar

search_bar()
