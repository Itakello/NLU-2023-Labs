import es_core_news_sm
import nltk
import pandas as pd
import sklearn_crfsuite
import spacy
from conll import evaluate
from nltk.corpus import conll2002
from sklearn_crfsuite import CRF, metrics
from spacy.tokenizer import Tokenizer


def sent2spacy_features(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)
    return feats

def sent2spacy_features_with_suffix(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_,
            'word[-3:]': token.text[-3:], # last 3 characters as the suffix
        }
        feats.append(token_feats)
    
    return feats

def sent2tutorial_features(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        features = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.text.isupper(),
            'word.istitle()': token.text.istitle(),
            'word.isdigit()': token.text.isdigit(),
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
        }
        feats.append(features)
    return feats

def sent2tutorial_features_1window(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for i, token in enumerate(spacy_sent):
        features = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.text.isupper(),
            'word.istitle()': token.text.istitle(),
            'word.isdigit()': token.text.isdigit(),
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
        }
        if i >= 1:
            token1 = spacy_sent[i-1]
            features.update({
                '-1:word.lower()': token1.lower_,
                '-1:word.istitle()': token1.text.istitle(),
                '-1:word.isupper()': token1.text.isupper(),
                '-1:word.isdigit()': token1.text.isdigit(),
                '-1:postag': token1.pos_,
                '-1:postag[:2]': token1.pos_[:2],
            })
        else:
            features['BOS'] = True
        if i <= len(sent)-2:
            token_1 = spacy_sent[i+1]
            features.update({
                '+1:word.lower()': token_1.lower_,
                '+1:word.istitle()': token_1.text.istitle(),
                '+1:word.isupper()': token_1.text.isupper(),
                '+1:word.isdigit()': token_1.text.isdigit(),
                '+1:postag': token_1.pos_,
                '+1:postag[:2]': token_1.pos_[:2],
            })
        else:
            features['EOS'] = True
        feats.append(features)
    return feats

def sent2tutorial_features_2window(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for i, token in enumerate(spacy_sent):
        features = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.text[-3:],
            'word[-2:]': token.text[-2:],
            'word.isupper()': token.text.isupper(),
            'word.istitle()': token.text.istitle(),
            'word.isdigit()': token.text.isdigit(),
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
        }
        if i >= 2:
            token2 = spacy_sent[i-2]
            features.update({
                '-2:word.lower()': token2.lower_,
                '-2:word.istitle()': token2.text.istitle(),
                '-2:word.isupper()': token2.text.isupper(),
                '-2:word.isdigit()': token2.text.isdigit(),
                '-2:postag': token2.pos_,
                '-2:postag[:2]': token2.pos_[:2],
            })
        if i >= 1:
            token1 = spacy_sent[i-1]
            features.update({
                '-1:word.lower()': token1.lower_,
                '-1:word.istitle()': token1.text.istitle(),
                '-1:word.isupper()': token1.text.isupper(),
                '-1:word.isdigit()': token1.text.isdigit(),
                '-1:postag': token1.pos_,
                '-1:postag[:2]': token1.pos_[:2],
            })
        if i < 2:
            features['BOS'] = True
        if i <= len(sent)-3:
            token_2 = spacy_sent[i+2]
            features.update({
                '+2:word.lower()': token_2.lower_,
                '+2:word.istitle()': token_2.text.istitle(),
                '+2:word.isupper()': token_2.text.isupper(),
                '+2:word.isdigit()': token_2.text.isdigit(),
                '+2:postag': token_2.pos_,
                '+2:postag[:2]': token_2.pos_[:2],
            })
        if i <= len(sent)-2:
            token_1 = spacy_sent[i+1]
            features.update({
                '+1:word.lower()': token_1.lower_,
                '+1:word.istitle()': token_1.text.istitle(),
                '+1:word.isupper()': token_1.text.isupper(),
                '+1:word.isdigit()': token_1.text.isdigit(),
                '+1:postag': token_1.pos_,
                '+1:postag[:2]': token_1.pos_[:2],
            })
        if i >= len(sent)-2:
            features['EOS'] = True
        feats.append(features)
    return feats

def word2features(sent, i):
    word = sent[i][0]
    return {'bias': 1.0, 'word.lower()': word.lower()}

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def train_cfr(X_train, y_train):
    crf = CRF(
        algorithm='lbfgs', 
        c1=0.1, 
        c2=0.1, 
        max_iterations=100, 
        all_possible_transitions=True
    )
    try:
        crf.fit(X_train, y_train)
    except AttributeError:
        pass
    return crf

def evaluate_cfr(y_test, hyp):
    results = evaluate(y_test, hyp)
    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl.round(decimals=3)
    print(pd_tbl)