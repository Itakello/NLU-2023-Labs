# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    mapping_spacy_to_NLTK = {
        "ADJ": "ADJ",
        "ADP": "ADP",
        "ADV": "ADV",
        "AUX": "VERB",
        "CCONJ": "CONJ",
        "DET": "DET",
        "INTJ": "X",
        "NOUN": "NOUN",
        "NUM": "NUM",
        "PART": "PRT",
        "PRON": "PRON",
        "PROPN": "NOUN",
        "PUNCT": ".",
        "SCONJ": "CONJ",
        "SYM": "X",
        "VERB": "VERB",
        "X": "X"
    }
    train_indx = math.ceil(len(treebank.tagged_sents()) * 0.8)
    trn_data = treebank.tagged_sents(tagset='universal')[:train_indx]
    tst_data = treebank.tagged_sents(tagset='universal')[train_indx:]
    train_and_evaluate_ngramtagger(trn_data, tst_data)
    evaluate_spacy_postagger(tst_data, mapping_spacy_to_NLTK)