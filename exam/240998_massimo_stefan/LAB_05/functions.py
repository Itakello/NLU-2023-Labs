import nltk
from nltk import PCFG, Nonterminal
from nltk.parse.generate import generate


def get_grammar() -> PCFG:
    weighted_rules = [
        'S -> NP VP [1.0]',
        'NP -> Det N [0.4] | Det N PP [0.4] | PRON [0.2]',
        'VP -> V NP [0.7] | V NP PP [0.3]',
        'PP -> P NP [1.0]',
        'Det -> "the" [0.2] | "a" [0.8]',
        'N -> "star" [0.2] | "castle" [0.2] | "hand" [0.2] | "man" [0.4]',
        'PRON -> "I" [0.5] | "You" [0.5]',
        'V -> "saw" [0.5] | "grab" [0.5]',
        'P -> "of" [0.5] | "on" [0.5]'
    ]
    my_grammar = nltk.PCFG.fromstring(weighted_rules)
    return my_grammar

def validate_grammar(grammar:PCFG, test_sents) -> None:
    parser = nltk.ViterbiParser(grammar)
    for sent in test_sents:
        parses = parser.parse(sent.split())
        print(len(list(parses)))
        
def generate_sentences(grammar:PCFG, n:int) -> None:
    for sent in generate(grammar, start=Nonterminal('S'), n=n):
        print(sent)