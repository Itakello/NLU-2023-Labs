# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math

import en_core_web_sm
from nltk.corpus import treebank
from nltk.lm.preprocessing import flatten
from nltk.metrics import accuracy
from nltk.tag import NgramTagger
from spacy.tokenizer import Tokenizer


def train_and_evaluate_ngramtagger(trn_data, tst_data) -> None:
    # training ngram tagger on treebank
    ngram_tagger = NgramTagger(3, trn_data)

    accuracy_ngram_tagger = ngram_tagger.accuracy(tst_data)
    print("Accuracy NgramTagger: {:.4f}".format(accuracy_ngram_tagger))
    
def evaluate_spacy_postagger(tst_data, mapping_spacy_to_NLTK) -> None:
    nlp = en_core_web_sm.load()
    # We overwrite the spacy tokenizer with a custom one, that split by whitespace only
    nlp.tokenizer = Tokenizer(nlp.vocab) # Tokenize by whitespace

    # Convert output to required format
    output = []
    for sent in tst_data:
        words = [word for word, tag in sent]
        tags = [tag for word, tag in sent]
        doc = nlp(' '.join(words))
        output.append([(token.text, mapping_spacy_to_NLTK[token.pos_]) for token in doc])

    # Flatten into a list
    flat_output = flatten(output)
    flat_tst_data = flatten(tst_data)

    # Evaluate using accuracy from nltk.metrics
    accuracy_pos_tagger = accuracy([tag for word, tag in flat_tst_data], [tag for word, tag in flat_output])
    print("Accuracy spacy POS-tagger: {:.4f}".format(accuracy_pos_tagger))