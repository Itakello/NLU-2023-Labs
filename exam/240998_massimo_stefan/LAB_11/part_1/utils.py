import nltk
from nltk.corpus import movie_reviews as mr
from nltk.corpus import subjectivity as sub
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

class SubjectivityDataset(data.Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=130):
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
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def split_dataset(sentences, labels, test_size=0.2, validation_size=0.1):
    # Encode labels
    labels = LabelEncoder().fit_transform(labels)
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=test_size, random_state=42)

    # Calculate the validation size as a proportion of the training set
    validation_size_adjusted = validation_size / (1 - test_size)

    # Split the train set into train and validation sets
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=validation_size_adjusted, random_state=42)
    return X_train, X_eval, X_test, y_train, y_eval, y_test

def get_sub_data():
    categories = sub.categories()
    sentences = []
    labels = []
    for cat in categories:
        tmp_sent = sub.sents(categories=cat)
        sentences.extend(tmp_sent)
        labels.extend([cat] * len(tmp_sent))
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return sentences, labels, encoder

def get_sub_dataloaders(sentences, labels):
    X_train, X_eval, X_test, y_train, y_eval, y_test = split_dataset(sentences, labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SubjectivityDataset(X_train, y_train, tokenizer, 100)
    eval_dataset = SubjectivityDataset(X_eval, y_eval, tokenizer, 100)
    test_dataset = SubjectivityDataset(X_test, y_test, tokenizer, 100)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, eval_loader, test_loader
    