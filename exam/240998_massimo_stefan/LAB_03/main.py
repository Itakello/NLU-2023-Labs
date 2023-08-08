from functions import *

if __name__ == "__main__":
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