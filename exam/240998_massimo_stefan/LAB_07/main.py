from functions import *
import nltk

if __name__ == "__main__":
    nltk.download('conll2002')
    trn_sents = list(conll2002.iob_sents('esp.train'))
    tst_sents = list(conll2002.iob_sents('esp.testa'))
    
    nlp = es_core_news_sm.load()
    nlp.tokenizer = Tokenizer(nlp.vocab)
    
    y_train = [sent2labels(s) for s in trn_sents]
    y_test = [sent2labels(s) for s in tst_sents]
    sents_test = [[(text, iob) for text, pos, iob in sent] for sent in tst_sents]
    
    sent2feature_funcs = [sent2spacy_features, sent2spacy_features_with_suffix, sent2tutorial_features, sent2tutorial_features_1window, sent2tutorial_features_2window]
    
    for func in sent2feature_funcs:
        X_train = [func(s, nlp) for s in trn_sents]
        X_test = [func(s, nlp) for s in tst_sents]
        cfr = train_cfr(X_train, y_train)
        pred = cfr.predict(X_test)
        hyp = [[(X_test[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]
        print(func.__name__)
        evaluate_cfr(sents_test, hyp)