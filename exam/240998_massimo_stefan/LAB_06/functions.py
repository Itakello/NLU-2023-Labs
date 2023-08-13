import spacy
from spacy_conll import ConllFormatter
import spacy_stanza
import nltk
nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
from nltk.parse import DependencyEvaluator
from nltk.parse.dependencygraph import DependencyGraph
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from collections import defaultdict


def get_parsers():
    spacy_parser = _get_spacy_parser()
    stanza_parser = _get_stanza_parser()
    return spacy_parser, stanza_parser

def _get_spacy_parser() -> Language:
    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")

    # Set up the conll formatter 
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"deprel": {"nsubj": "subj"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)
    # Split by white space
    nlp.tokenizer = Tokenizer(nlp.vocab)
    return nlp

def _get_stanza_parser() -> Language:
    nlp = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"deprel": {"nsubj": "subj"}}}#, "root":"ROOT"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)
    return nlp

def get_parsed_sentences() :
    return dependency_treebank.parsed_sents()[-100:]

def get_raw_sentences():
    sentences = dependency_treebank.sents()[-100:]
    return [" ".join(sentence) for sentence in sentences]

def compare_dependency_tags(sentence: str, spacy_parser: Language, stanza_parser: Language) -> None:
    # Parse the sentence with spaCy
    spacy_doc = spacy_parser(sentence)
    spacy_tags = [(token.text, token.dep_) for token in spacy_doc]

    # Parse the sentence with Stanza
    stanza_doc = stanza_parser(sentence)
    stanza_tags = [(token.text, token.dep_) for token in stanza_doc]

    # Compare the tags
    print("Comparing dependency tags:")
    print(f"Sentence: {sentence}")
    print("spaCy tags:")
    print(spacy_tags)
    print("Stanza tags:")
    print(stanza_tags)

    if spacy_tags == stanza_tags:
        print("The dependency tags are the same for both parsers.")
    else:
        print("The dependency tags are different between the two parsers.")

def parse_sentences(parser:Language, sentences):
    parsed_sentences = []
    for sentence in sentences:
        doc = parser(sentence)
        # Convert doc to a pandas object
        df = doc._.pandas
        
        # the conversion_maps option in the config is not working
        df['DEPREL'].replace({'nsubj': 'subj', 'root': 'ROOT'}, inplace=True)

        # Select the columns accoroding to Malt-Tab format
        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        dp = DependencyGraph(tmp)
        parsed_sentences.append(dp)
    return parsed_sentences

def evaluate_parser(parsed_sentences) -> None:
    gold_sentences = get_parsed_sentences()
    #parsed_sentences[0].tree().pretty_print(unicodelines=True, nodedist=4)
    #gold_sentences[0].tree().pretty_print(unicodelines=True, nodedist=4)

    de = DependencyEvaluator(parsed_sentences, gold_sentences)
    las, uas = de.eval()
    print(f"LAS: {las}")
    print(f"UAS: {uas}")