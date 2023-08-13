import math
import nltk
import en_core_web_sm
from nltk.corpus import treebank
from nltk.lm.preprocessing import flatten
from nltk.metrics import accuracy
from nltk.tag import NgramTagger
from spacy.tokenizer import Tokenizer
from itertools import chain

def create_dataset():
    nltk.download('treebank')
    nltk.download('universal_tagset')
    train_indx = math.ceil(len(treebank.tagged_sents()) * 0.8)
    trn_data = treebank.tagged_sents(tagset='universal')[:train_indx]
    tst_data = treebank.tagged_sents(tagset='universal')[train_indx:]
    return trn_data, tst_data

def train_and_evaluate_ngramtagger(trn_data, tst_data, n, cutoff) -> None:
    # training ngram tagger on treebank
    print("\nNgram Tagger with n =", n, " and cutoff =", cutoff)
    ngram_tagger = NgramTagger(n, train=trn_data, cutoff=cutoff)

    accuracy_ngram_tagger = ngram_tagger.accuracy(tst_data)
    print("Accuracy NgramTagger: {:.4f}".format(accuracy_ngram_tagger))

def get_spacy_pos_tags(sentences):
    nlp = en_core_web_sm.load()
    pos_tags = []
    
    for sentence in sentences:
        tokens = [token for token, _ in sentence]
        doc = nlp(" ".join(tokens))
        tags = [token.tag_ for token in doc]
        pos_tags.append(list(zip(tokens, tags)))
    
    return pos_tags

def calculate_accuracy(true_tags, predicted_tags):
    correct = 0
    total = len(true_tags)
    
    for true_tag, predicted_tag in zip(true_tags, predicted_tags):
        if true_tag == predicted_tag:
            correct += 1
    accuracy = correct / total
    
    return accuracy

def map_spacy_tags_to_nltk(tags, mapping):
    mapped_tags = []
    
    for sentence_tags in tags:
        for token, tag in sentence_tags:
          if tag in mapping:
            mapped_tags.append((token, mapping[tag]))
          else:
            mapped_tags.append((token, 'X'))
    
    return mapped_tags
    
def evaluate_spacy_postagger(tst_data, mapping) -> None:
    spacy_pos_tags = get_spacy_pos_tags(tst_data)
    nltk_pos_tags = map_spacy_tags_to_nltk(spacy_pos_tags, mapping)
    spacy_pos_tags = list(chain.from_iterable(spacy_pos_tags)) # Flatten the list of lists 

    true_tags = [tag for sentence in tst_data for _, tag in sentence]

    spacy_predicted_tags = [tag for _, tag in spacy_pos_tags]
    nltk_predicted_tags = [tag for _, tag in nltk_pos_tags]

    spacy_accuracy = calculate_accuracy(true_tags, spacy_predicted_tags)
    print()
    print("Spacy Accuracy:", spacy_accuracy)

    nltk_accuracy = accuracy(true_tags, nltk_predicted_tags)
    print("NLTK Accuracy:", nltk_accuracy)
    
    return spacy_accuracy, nltk_accuracy