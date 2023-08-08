from functions import *

if __name__ == "__main__":
    data_raw, target = fetch_20newsgroups(subset='all', return_X_y=True)
    vectorizers = get_vectorizers()
    test_vectorizers(vectorizers, data_raw, target)