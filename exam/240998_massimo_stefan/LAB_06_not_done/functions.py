# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import spacy
import spacy_conll
import spacy_stanza
import stanza
from nltk.corpus import dependency_treebank
from nltk.parse import DependencyEvaluator
from nltk.parse.dependencygraph import DependencyGraph
from spacy.language import Language
from spacy.tokenizer import Tokenizer


def evaluate_parser(parser:Language, parsed_sentences:list[DependencyGraph]) -> None:
    gold_sentences = get_sentences()
    parsed_sentences[0].tree().pretty_print(unicodelines=True, nodedist=4)
    gold_sentences[0].tree().pretty_print(unicodelines=True, nodedist=4)

    de = DependencyEvaluator(parsed_sentences, gold_sentences)
    las, uas = de.eval()
    print(f"Parser: {parser}")
    print(f"LAS: {las}")
    print(f"UAS: {uas}")
    
def parse_sentences(parser:Language, sentences:list[str]) -> list[DependencyGraph]:
    parsed_sentences = []
    for sentence in sentences:
        doc = parser(sentence)
        # Convert doc to a pandas object
        df = doc._.pandas

        # Select the columns accoroding to Malt-Tab format
        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        dp = DependencyGraph(tmp)
        parsed_sentences.append(dp)
    return parsed_sentences

def get_sentences() -> list[DependencyGraph]:
    return dependency_treebank.parsed_sents()[-100:-99]

def get_parsers() -> list[Language]:
    spacy_parser = _get_spacy_parser()
    stanza_parser = _get_stanza_parser()
    return [spacy_parser, stanza_parser]

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
            "conversion_maps": {"deprel": {"nsubj": "subj", "root":"ROOT"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)
    return nlp