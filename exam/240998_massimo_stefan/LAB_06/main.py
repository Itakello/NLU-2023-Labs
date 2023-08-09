from functions import *

if __name__ == "__main__":
    spacy_parser, stanza_parser = get_parsers()
    compare_dependency_tags("The quick brown fox jumps over the lazy dog.", spacy_parser, stanza_parser)
    raw_sentences = get_raw_sentences()
    
    print("Parsing sentences with spaCy:")
    parsed_sentences = parse_sentences(spacy_parser, raw_sentences)
    evaluate_parser(parsed_sentences)
    print("Parsing sentences with stanza:")
    parsed_sentences = parse_sentences(stanza_parser, raw_sentences)
    evaluate_parser(parsed_sentences)