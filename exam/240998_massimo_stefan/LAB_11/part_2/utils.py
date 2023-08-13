import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
from torch.utils.data import DataLoader


def extract_data_from_file(file_name):
    file_path = os.path.join(os.path.dirname(__file__),'dataset',file_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    sentences = []
    annotations = []
    for line in lines:
        sentence, annotation = line.strip().split("####")
        sentences.append(sentence)
        tags = [(ann.split('=')[0], ann.split('=')[1]) for ann in annotation.split()]
        annotations.append(tags)
    return sentences, annotations

class ABSADataset(Dataset):
    def __init__(self, sentences, annotations, tokenizer, max_len):
        self.sentences = sentences
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        annotations = self.annotations[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Align the annotations with the tokens, since a word can be splitted in multiple tokens
        aligned_annotations = self._align_annotations(sentence, annotations)
        
        # Conversion + padding
        at_annotations = [0 if ann == 'O' else 1 for ann in aligned_annotations] + [0] * (self.max_len - len(aligned_annotations))
        
        po_mask = {
            'O':0,
            'T-POS':1,
            'T-NEG':2,
            'T-NEU':3
        }
        # Conversion + padding
        po_annotations = [po_mask[ann] for ann in aligned_annotations] + [0] * (self.max_len - len(aligned_annotations))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'at_annotations': torch.tensor(at_annotations, dtype=torch.long),
            'po_annotations': torch.tensor(po_annotations, dtype=torch.long)
        }
        
    def _align_annotations(self, sentence, annotations):
        aligned_annotations = []
        tokens = self.tokenizer.tokenize(sentence)
        
        special_characters = ['(', ')', "'", "/", "\\", "-", '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '"', '*']
        
        word_idx = 0
        token_idx = 0
        
        while token_idx < len(tokens) and word_idx < len(annotations):
            token = tokens[token_idx]
            word, annotation = annotations[word_idx]
            
            # Exact match
            if token == word.lower():
                aligned_annotations.append(annotation)
                word_idx += 1
                token_idx += 1
            # Special characters
            elif token in special_characters:
                aligned_annotations.append('O')
                token_idx += 1
            # Partial match (subword tokens)
            elif word.lower().startswith(token.replace("##", "")):
                aligned_annotations.append(annotation)
                word = word[len(token.replace("##", "")):]
                token_idx += 1
            else:
                word_idx += 1

        # Handle any remaining tokens
        while token_idx < len(tokens):
            token = tokens[token_idx]
            if token in special_characters:
                aligned_annotations.append('O')
            else:
                aligned_annotations.append('O')  # Default to 'O' for any unhandled cases
            token_idx += 1

        assert len(tokens) == len(aligned_annotations), "Tokens and annotations length mismatch!"
        
        return aligned_annotations

def get_dataloaders(batch_size=32, max_len=100):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_sentences, train_annotations = extract_data_from_file('train.txt')
    # since there is not dev.txt, split the train set into train and dev
    train_sentences, val_sentences = train_sentences[:int(len(train_sentences)*0.8)], train_sentences[int(len(train_sentences)*0.8):]
    train_annotations, val_annotations = train_annotations[:int(len(train_annotations)*0.8)], train_annotations[int(len(train_annotations)*0.8):]
    
    test_sentences, test_annotations = extract_data_from_file('test.txt')
    
    train_dataset = ABSADataset(train_sentences, train_annotations, tokenizer, max_len)
    test_dataset = ABSADataset(test_sentences, test_annotations, tokenizer, max_len)
    val_dataset = ABSADataset(val_sentences, val_annotations, tokenizer, max_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, val_dataloader