# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from collections import Counter

import nltk
import numpy as np
from nltk import pos_tag
from nltk.corpus import senseval, stopwords, wordnet, wordnet_ic
from nltk.corpus.reader import SensevalInstance
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.wsd import lesk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

semcor_ic = wordnet_ic.ic('ic-semcor.dat')

def get_vectors_coll_features_extended(instances):
    data_col = [collocational_features(inst) for inst in instances]
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(data_col)

def get_vectors_bow(instances):
    data = [" ".join([t[0] for t in inst.context]) for inst in instances]
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data)

def get_vectors_bow_and_coll_features_extended(instances):
    vectors_coll_features_extended = get_vectors_coll_features_extended(instances)
    vectors_bow = get_vectors_bow(instances)
    new_vectors = np.concatenate((vectors_coll_features_extended, vectors_bow.toarray()), axis=1)
    return new_vectors
     
def get_labels():
    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]
    lblencoder = LabelEncoder()
    lblencoder.fit(lbls)
    return lblencoder.transform(lbls)

def get_pos_features(instance: SensevalInstance):
    context = [t[0] for t in instance.context]
    pos = pos_tag(context)
    return pos[instance.position][1]

def get_ngram_features(instance: SensevalInstance, n=3):
    context = [t[0] for t in instance.context]
    position = instance.position
    start_index = max(0, position - n + 1)
    end_index = min(len(context), position + n)
    ngram_tuples = list(ngrams(context[start_index:end_index], n))
    return [' '.join(ngram) for ngram in ngram_tuples]

def collocational_features(inst):
    p = inst.position
    return {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        'pos': get_pos_features(inst),
        'ngram': get_ngram_features(inst, 3),
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0]
    }

def evaluate_vectors(vectors, stratified_split, labels):
    classifier = MultinomialNB()
    labels = get_labels()
    scores = cross_validate(classifier, vectors, labels, cv=stratified_split, scoring=['f1_micro'])
    #print(sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))
    print(np.mean(scores['test_f1_micro']))
    
#--------------------------------Part 2--------------------------------

def preprocess(text):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = stopwords.words('english')
    
    lem = WordNetLemmatizer()
    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    # compute pos-tag
    tagged = nltk.pos_tag(tokens, tagset="universal")
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    # unique the list
    tagged = list(set(tagged))
    
    return tagged

def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))    
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense

def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)
    
    # let's get senses for each
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]

    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words 
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions

def get_top_sense(words, sense_list):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense

def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):
    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []

    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))

    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        # We remove 0 scores, senses without overlapping
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense

def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, 
                    synsets=None, majority=True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    
    # Here you may have some room for improvement
    # For instance instead of using all the definitions from the context
    # you pick the most common one of each word (i.e. the first)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense

def pedersen_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, 
                    synsets=None, threshold=0.1):
    
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    synsets_scores = {}
    for ss_tup in synsets:
        ss = ss_tup[0]
        if ss not in synsets_scores:
            synsets_scores[ss] = 0
        for senses in context_senses:
            scores = []
            for sense in senses[1]:
                if similarity == "path":
                    try:
                        scores.append((sense[0].path_similarity(ss), ss))
                    except:
                        scores.append((0, ss))    
                elif similarity == "lch":
                    try:
                        scores.append((sense[0].lch_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "wup":
                    try:
                        scores.append((sense[0].wup_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "resnik":
                    try:
                        scores.append((sense[0].res_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "lin":
                    try:
                        scores.append((sense[0].lin_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "jiang":
                    try:
                        scores.append((sense[0].jcn_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                else:
                    print("Similarity metric not found")
                    return None
            value, sense = max(scores)
            if value > threshold:
                synsets_scores[sense] = synsets_scores[sense] + value
    
    values = list(synsets_scores.values())
    if sum(values) == 0:
        print('Warning all the scores are 0')
    senses = list(synsets_scores.keys())
    best_sense_id = values.index(max(values))
    return senses[best_sense_id]

def get_lesk_predictions(instances, lesk_method):
    preds = []
    for inst in instances:
        ambiguous_word = inst.context[inst.position][0]
        context = " ".join([t[0] for t in inst.context])
        sense = lesk_method(context, ambiguous_word).name()
        preds.append(sense)
    return preds
    
def evaluate_lesk(instances, lesk_func, stratified_split, mapping):
    f1_scores = []
    for _, test_index in stratified_split.split(instances, np.zeros(len(instances))):
        test_instances = [instances[i] for i in test_index]

        # Predict senses for test instances
        predicted_senses = get_lesk_predictions(test_instances, lesk_func)
        real_senses = [mapping[inst.senses[0]] for inst in test_instances]

        f1_scores.append(f1_score(real_senses, predicted_senses, average='micro'))

    print(np.mean(f1_scores))