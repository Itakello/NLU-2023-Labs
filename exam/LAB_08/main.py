from functions import *

if __name__ == "__main__":
    instances = senseval.instances('interest.pos')
    
    vectors_coll_features_extended = get_vectors_coll_features_extended(instances)
    vectors_bow_and_coll_features_extended = get_vectors_bow_and_coll_features_extended(instances)
    
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    labels = get_labels()
    
    print('Evaluating vectors_coll_features_extended')
    evaluate_vectors(vectors_coll_features_extended, stratified_split, labels)
    print('Evaluating vectors_bow_and_coll_features_extended')
    evaluate_vectors(vectors_bow_and_coll_features_extended, stratified_split, labels)
    
    """mapping = {
        'interest_1': 'interest.n.01',
        'interest_2': 'interest.n.03',
        'interest_3': 'pastime.n.01',
        'interest_4': 'sake.n.01',
        'interest_5': 'interest.n.05',
        'interest_6': 'interest.n.04',
    }"""
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