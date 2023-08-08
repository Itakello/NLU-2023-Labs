from functions import *

if __name__ == "__main__":
    parsers = get_parsers()
    #compare_dependency_tags("The quick brown fox jumps over the lazy dog.", parsers[0], parsers[1])
    sentences = get_sentences()
    # Convert the tokenized sentences back to raw text
    raw_sentences = [" ".join(token["word"] for token in sent.nodes.values() if token["word"] is not None) for sent in sentences]
    for parser in parsers:
        parsed_sentences = parse_sentences(parser, raw_sentences)
        evaluate_parser(parser, parsed_sentences)