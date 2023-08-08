# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    m_chars, m_words, m_sents = get_corpus()
    print_statistics(m_chars, m_words, m_sents)
    print("--Statistics using Spacy--")
    chars_spacy, words_spacy, sentes_spacy = get_stats_spacy(m_chars)
    print_statistics(chars_spacy, words_spacy, sentes_spacy)
    chars_nltk, words_nltk, sentes_nltk = get_stats_nltk(m_chars)
    print("--Statistics using NLTK--")
    print_statistics(chars_nltk, words_nltk, sentes_nltk)
    print("--Lexicon sizes for each version--")
    compute_low_lexicons(m_words, words_spacy, words_nltk)
    print("--Frequency diustribution for each version--")
    compute_freq_distributions(m_words, words_spacy, words_nltk)