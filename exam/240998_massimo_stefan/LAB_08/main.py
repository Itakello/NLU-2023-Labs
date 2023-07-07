# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

"""
**Same test set for all the experiments, you can use K-fold validation**
- Extend collocational features with
    - POS-tags
    - Ngrams within window
- Concatenate BOW and new collocational feature vectors & evaluate
- Evaluate Lesk Original and Graph-based (Lesk Similarity or Pedersen) metrics on the same test split and compare
"""

from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    instances = senseval.instances('interest.pos')
    
    vectors_coll_features_extended = get_vectors_coll_features_extended(instances)
    vectors_bow_and_coll_features_extended = get_vectors_bow_and_coll_features_extended(instances)
    
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    labels = get_labels()
    
    print('Evaluating vectors_coll_features_extended')
    evaluate_vectors(vectors_coll_features_extended, stratified_split, labels)
    print('Evaluating vectors_bow_and_coll_features_extended')
    evaluate_vectors(vectors_bow_and_coll_features_extended, stratified_split, labels)
    
    mapping = {
        'interest_1': 'interest.n.01',
        'interest_2': 'interest.n.02',
        'interest_3': 'pastime.n.01',
        'interest_4': 'sake.n.01',
        'interest_5': 'interest.n.05',
        'interest_6': 'interest.n.06',
    }
    
    print('Evaluating Original Lesk')
    evaluate_lesk(instances, original_lesk, stratified_split, mapping)
    print('Evaluating Graph-based Lesk-similarity')
    evaluate_lesk(instances, lesk_similarity, stratified_split, mapping)
    print('Evaluating Graph-based Pedersen-similarity')
    evaluate_lesk(instances, pedersen_similarity, stratified_split, mapping)