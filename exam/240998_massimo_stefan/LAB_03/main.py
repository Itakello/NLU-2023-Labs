# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    dataset, lex = get_dataset_and_voc()
    NGRAM_LENGTHS = 3
    test_sents = [
        ['to', 'be', 'or', 'not', 'to', 'be'],
        ['out', 'damned', 'spot'],
        ['is', 'this', 'a', 'dagger', 'which', 'i', 'see', 'before', 'me'],
        ['double', 'double', 'toil', 'and', 'trouble'],
        ['something', 'wicked', 'this', 'way', 'comes']
    ]
    evaluate_base_stupidbackoff(NGRAM_LENGTHS, dataset, lex, test_sents)
    evaluate_my_stupidbackoff(NGRAM_LENGTHS, dataset, lex, test_sents)