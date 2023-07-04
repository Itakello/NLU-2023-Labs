# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC


def get_vectorizers() -> dict:
    stop_words = stopwords.words('english')
    return {"CountVect":CountVectorizer(binary=True), "TF-IDF [CutOff]":TfidfVectorizer(min_df=2, max_df=100), "TF-IDF [WithoutStopWords]":TfidfVectorizer(stop_words=stop_words), "TF-IDF [NoLowercase]":TfidfVectorizer(lowercase=False)}

def test_vectorizers(vectorizers: dict, data_raw, target, C = 1e-2) -> None:
    clf = LinearSVC(C=C, dual=True)
    stratified_split = StratifiedKFold(n_splits=10, shuffle=True)

    for experiment_id, vectorizer in vectorizers.items():
        vectorized_data = vectorizer.fit_transform(data_raw)
        scores = cross_validate(clf, vectorized_data, target, cv=stratified_split, scoring=['f1_macro'])
        final_score = round(sum(scores['test_f1_macro'])/len(scores['test_f1_macro']),2)
        print(experiment_id, final_score)