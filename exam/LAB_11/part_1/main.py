from functions import *
from model import *
from utils import *

if __name__ == "__main__":  
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    criterion = nn.CrossEntropyLoss()
    
    sub_model = 'subjectivity_bert'
    
    sub_sentences, sub_labels = get_sub_sent()
    
<<<<<<< HEAD:exam/240998_massimo_stefan/LAB_11/part_1/main.py
    print('Subjectivity task: full dataset')
    k_fold_evaluation(criterion, tokenizer, sub_sentences, sub_labels, sub_model, n_splits=2)
=======
    #k_fold_evaluation(criterion, tokenizer, sub_sentences, sub_labels, sub_model, batch_size=64, n_splits=10)
>>>>>>> d502da6e1bc16356521b6a2ac2dcb7344b05a3f2:exam/LAB_11/part_1/main.py
    
    mr_documents, mr_labels = get_mr_doc()
    
    print('Polarity task: full dataset')
    k_fold_evaluation(criterion, tokenizer, mr_documents, mr_labels, 'pol_bert_full', batch_size=16, n_splits=10)
    
    mr_documents_subj, mr_labels_subj = filter_subj_doc(mr_documents, mr_labels, tokenizer, sub_model)
    
    print('Polarity task: subjective dataset')
    k_fold_evaluation(criterion, tokenizer, mr_documents_subj, mr_labels_subj, 'pol_bert_subj', batch_size=16, n_splits=10)