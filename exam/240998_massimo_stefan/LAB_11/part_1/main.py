from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    sub_model = 'subjectivity_bert'
    
    sub_sentences, sub_labels = get_sub_sent()
    
    k_fold_evaluation(criterion, tokenizer, sub_sentences, sub_labels, sub_model, n_splits=2)
    
    mr_documents, mr_labels = get_mr_doc()
    
    print('Polarity task: full dataset')
    k_fold_evaluation(criterion, tokenizer, mr_documents, mr_labels, 'pol_bert_full', n_splits=2)
    
    mr_documents_subj, mr_labels_subj = filter_subj_doc(mr_documents, tokenizer, sub_model)
    
    print('Polarity task: subjective dataset')
    k_fold_evaluation(criterion, tokenizer, mr_documents, mr_labels, 'pol_bert_subj', n_splits=2)