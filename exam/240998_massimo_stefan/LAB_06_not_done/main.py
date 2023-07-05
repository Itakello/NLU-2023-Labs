# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    parsers = get_parsers()
    sentences = get_sentences()
    # Convert the tokenized sentences back to raw text
    raw_sentences = [" ".join(token["word"] for token in sent.nodes.values() if token["word"] is not None) for sent in sentences]
    for parser in parsers:
        parsed_sentences = parse_sentences(parser, raw_sentences)
        evaluate_parser(parser, parsed_sentences)