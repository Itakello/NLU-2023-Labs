from functions import *

if __name__ == "__main__":
    parsers = get_parsers()
    sentences = get_sentences()
    # Convert the tokenized sentences back to raw text
    raw_sentences = [" ".join(token["word"] for token in sent.nodes.values() if token["word"] is not None) for sent in sentences]
    for parser in parsers:
        parsed_sentences = parse_sentences(parser, raw_sentences)
        evaluate_parser(parser, parsed_sentences)