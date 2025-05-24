from indexing import load_index
from preprocessing import stemming
import os
import re

cwd = os.getcwd()

inverted_index = load_index(os.path.join(cwd, "Indexes/inverted_index.json"))
positional_index = load_index(os.path.join(cwd, "Indexes/positional_index.json"))


# --------------------------------------------------------------------------------------------------------------------------------------------
# Boolean Query Handler
# ------------------------
def get_docs(term):
    return set(inverted_index.get(term, []))


def boolean_query_parser(tokens):
    def eval_not(op):
        all_docs = set(doc_id for doc_list in inverted_index.values() for doc_id in doc_list)
        return all_docs - op

    def eval_and(op1, op2):
        return op1 & op2

    def eval_or(op1, op2):
        return op1 | op2

    precedence = {"NOT": 3, "AND": 2, "OR": 1, "(": 0, ")": 0}
    output = []
    operators = []

    for token in tokens:
        if token not in precedence:  # It's a term
            output.append(get_docs(token))
        elif token == "(":  # Left parenthesis
            operators.append(token)
        elif token == ")":  # Right parenthesis
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            operators.pop()  # Remove "("
        else:
            while operators and precedence[operators[-1]] >= precedence[token]:
                output.append(operators.pop())
            operators.append(token)

    while operators:
        output.append(operators.pop())

    stack = []

    for token in output:
        if isinstance(token, set):
            stack.append(token)
        elif token == "NOT":
            operand = stack.pop()
            stack.append(eval_not(operand))
        else:  # AND / OR
            operand2 = stack.pop()
            operand1 = stack.pop()
            if token == "AND":
                stack.append(eval_and(operand1, operand2))
            elif token == "OR":
                stack.append(eval_or(operand1, operand2))

    return stack[0] if stack else set()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Proximity Queries handling
# --------------------------
def process_proximity_query(query):
    match = re.search(r"(\w+)\s+(\w+)\s*/(\d+)", query)
    if not match:
        return set()

    word1, word2, distance = match.groups()

    word1, word2 = stemming(word1), stemming(word2)
    distance = int(distance)

    return proximity_search(word1, word2, distance)


def proximity_search(word1, word2, max_distance):
    if word1 not in positional_index or word2 not in positional_index:
        return set()

    docs_with_word1 = set(positional_index[word1].keys())
    docs_with_word2 = set(positional_index[word2].keys())
    common_docs = docs_with_word1 & docs_with_word2  # Only check docs containing both words

    result_docs = set()

    for doc in common_docs:
        positions1 = positional_index[word1][doc]
        positions2 = positional_index[word2][doc]

        for pos1 in positions1:
            for pos2 in positions2:
                if abs(pos1 - pos2) <= max_distance:
                    result_docs.add(doc)
                    break  # No need to check more positions

    return result_docs


# -----------------------------------------------------------------------------------------------------------

def stem_query(query):
    tokens = re.findall(r'\w+|AND|OR|NOT|\(|\)', query.upper())
    return [stemming(token) if token not in ["AND", "OR", "NOT", "(", ")"] else token for token in tokens]


def is_proximity_query(query):
    return bool(re.search(r"\w+\s+\w+\s*/\d+", query))


# Main query handler, using above 2 helper funcs
# ------------------------
def process_query(query):
    # We first check the query type (boolean operator or proximity query then call functions accordingly)
    if is_proximity_query(query):
        result_docs = process_proximity_query(query)
    else:
        stemmed_tokens = stem_query(query)
        result_docs = boolean_query_parser(stemmed_tokens)

    return sorted(map(int, result_docs))