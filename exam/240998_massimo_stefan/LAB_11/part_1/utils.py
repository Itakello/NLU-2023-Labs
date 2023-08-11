from nltk.corpus import movie_reviews as mr
from nltk.corpus import subjectivity as sub
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import re
from nltk.tokenize import sent_tokenize

class SentimentDataset(data.Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_loader(sentences, labels, tokenizer, shuffle=False):
    dataset = SentimentDataset(sentences, labels, tokenizer)
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)

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