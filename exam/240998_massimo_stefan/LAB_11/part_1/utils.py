from nltk.corpus import movie_reviews as mr
from nltk.corpus import subjectivity as sub
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import re
from nltk.tokenize import sent_tokenize
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = 0

class SentimentDataset(data.Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            padding=False
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
def collate_fn(batch):
    # Sort data by sequence lengths
    batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
    
    sentences = [item['sentence_text'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Pad sequences to the length of the longest sequence in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=PAD_TOKEN)
    
    return {
        'sentence_texts': sentences,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': torch.stack(labels)
    }

def create_loader(sentences, labels, tokenizer, shuffle=False):
    dataset = SentimentDataset(sentences, labels, tokenizer)
    return DataLoader(dataset, batch_size=64, shuffle=shuffle, collate_fn=collate_fn)

def get_sub_sent():
    categories = sub.categories()
    sentences = []
    labels = []
    for cat in categories:
        tmp_sents = sub.sents(categories=cat)
        tmp_sents = [' '.join(tmp_sent) for tmp_sent in tmp_sents]
        sentences.extend(tmp_sents)
        labels.extend([cat] * len(tmp_sents))
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return sentences, labels

def custom_sent_tokenize(text):
    sentences = sent_tokenize(text)
    filtered_sentences = []
    for sentence in sentences:
        # Filter out sentences that are too short
        if len(sentence) < 3:
            continue
        # Filter out sentences that contain only non-alphanumeric characters
        if re.match(r'^\W+$', sentence):
            continue
        filtered_sentences.append(sentence)
    return filtered_sentences

def get_mr_doc():
    categories = mr.categories()
    documents = []
    labels = []
    for cat in categories:
        fileids = mr.fileids(categories=cat)
        tmp_documents = [mr.raw(fileid) for fileid in fileids]
        documents.extend(tmp_documents)
        labels.extend([cat] * len(documents))
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return documents, labels