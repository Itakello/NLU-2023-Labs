from functions import *

if __name__ == "__main__":
    test_sents = ["I grab the hand of a star", "You saw a man on a castle"]
    grammar = get_grammar()
    
    validate_grammar(grammar, test_sents)
    generate_sentences(grammar, 10)