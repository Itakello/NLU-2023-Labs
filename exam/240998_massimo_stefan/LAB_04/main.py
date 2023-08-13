from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    mapping_spacy_to_NLTK = {
        "NN": "NN",
        "NNS": "NNS",
        "NNP": "NNP",
        "NNPS": "NNPS",
        "VB": "VB",
        "VBD": "VBD",
        "VBG": "VBG",
        "VBN": "VBN",
        "VBP": "VBP",
        "VBZ": "VBZ",
        "JJ": "JJ",
        "JJR": "JJR",
        "JJS": "JJS",
        "RB": "RB",
        "RBR": "RBR",
        "RBS": "RBS",
        "IN": "IN",
        "DT": "DT",
        "PDT": "PDT",
        "CC": "CC",
        "CD": "CD",
        ".": ".",
        ",": ",",
        ":": ":",
        ";": ":",
        "\"": ".",
        "'": ".",
        "-LRB-": "-LRB-",
        "-RRB-": "-RRB-",
        "-LSB-": "-LRB-",
        "-RSB-": "-RRB-",
        "-LCB-": "-LRB-",
        "-RCB-": "-RRB-"
    }
    trn_data, tst_data = create_dataset()
    train_and_evaluate_ngramtagger(trn_data, tst_data, 2, 2)
    train_and_evaluate_ngramtagger(trn_data, tst_data, 4, 1)
    train_and_evaluate_ngramtagger(trn_data, tst_data, 2, 1)
    train_and_evaluate_ngramtagger(trn_data, tst_data, 1, 1)
    evaluate_spacy_postagger(tst_data, mapping_spacy_to_NLTK)