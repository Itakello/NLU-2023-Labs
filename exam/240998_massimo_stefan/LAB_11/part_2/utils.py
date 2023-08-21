import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = 0

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
    def __init__(self, sentences, annotations, tokenizer, lang):
        self.sentences = sentences
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.lang = lang
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        annotations = self.annotations[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Align the annotations with the tokens, since a word can be splitted in multiple tokens
        aligned_annotations = self._align_annotations(sentence, annotations)
        length = len(aligned_annotations)
        
        aspect_annotations = [PAD_TOKEN] + [self.lang.aspect2idx[ann[0]] for ann in aligned_annotations] + [PAD_TOKEN]
        
        polarity_annotations = [PAD_TOKEN] + [self.lang.polarity2idx[ann] for ann in aligned_annotations] + [PAD_TOKEN]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'length': length,
            'attention_mask': encoding['attention_mask'].flatten(),
            'aspect_annotations': torch.tensor(aspect_annotations, dtype=torch.long),
            'polarity_annotations': torch.tensor(polarity_annotations, dtype=torch.long)
        }
        
    def _align_annotations(self, sentence, annotations):
        aligned_annotations = []
        tokens = self.tokenizer.tokenize(sentence)

        # Create a pointer for current position in the original annotations
        annotation_pointer = 0
        
        # Iterate through each token
        for token in tokens:
            # Check if the current token matches the start of the current word in annotations
            if annotation_pointer < len(annotations) and annotations[annotation_pointer][0].lower().startswith(token.replace("##", "")):
                aligned_annotations.append(annotations[annotation_pointer][1])
                # If the token matches the entire word in annotations, move to next word
                if token.replace("##", "") == annotations[annotation_pointer][0].lower():
                    annotation_pointer += 1
            # If the token is a special character or subword token and does not match the start of the word in annotations
            else:
                # Add the previous word's annotation if it exists
                if annotation_pointer > 0:
                    aligned_annotations.append(annotations[annotation_pointer - 1][1])
                # Otherwise, consider it as an 'O' (Outside) annotation
                else:
                    aligned_annotations.append('O')

        assert len(tokens) == len(aligned_annotations), f"Tokens and annotations length mismatch! Tokens: {tokens} Annotations: {annotations} Aligned annotations: {aligned_annotations}"

        return aligned_annotations

class Lang():
    def __init__(self) -> None:
        self.aspect2idx = {'O':1, 'T':2}
        self.polarity2idx = {'O':1, 'T-POS':2, 'T-NEG':3, 'T-NEU':4}
        self.idx2aspect = {v:k for k,v in self.aspect2idx.items()}
        self.idx2polarity = {v:k for k,v in self.polarity2idx.items()}

def collate_fn(batch):
    # Sort data by sequence lengths
    batch.sort(key=lambda x: len(x['input_ids']), reverse=True)
    
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    aspect_annotations = [item['aspect_annotations'] for item in batch]
    polarity_annotations = [item['polarity_annotations'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Pad sequences to the length of the longest sequence in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=PAD_TOKEN)
    aspect_annotations = pad_sequence(aspect_annotations, batch_first=True, padding_value=PAD_TOKEN)
    polarity_annotations = pad_sequence(polarity_annotations, batch_first=True, padding_value=PAD_TOKEN)
    
    return {
        'input_ids': input_ids,
        'lengths': torch.tensor(lengths, dtype=torch.long),
        'attention_mask': attention_masks,
        'aspect_annotations': aspect_annotations,
        'polarity_annotations': polarity_annotations
    }

def get_dataloaders(lang):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_sentences, train_annotations = extract_data_from_file('train.txt')
    
    # since there is not dev.txt, split the train set into train and dev
    train_sentences, val_sentences = train_sentences[:int(len(train_sentences)*0.8)], train_sentences[int(len(train_sentences)*0.8):]
    train_annotations, val_annotations = train_annotations[:int(len(train_annotations)*0.8)], train_annotations[int(len(train_annotations)*0.8):]
    
    test_sentences, test_annotations = extract_data_from_file('test.txt')
    
    train_dataloader = create_loader(train_sentences, train_annotations, tokenizer, lang, shuffle=True)
    test_dataloader = create_loader(test_sentences, test_annotations, tokenizer, lang)
    val_dataloader = create_loader(val_sentences, val_annotations, tokenizer, lang)
    
    return train_dataloader, test_dataloader, val_dataloader

def create_loader(sentences, annotations, tokenizer, lang, shuffle=False):
    dataset = ABSADataset(sentences, annotations, tokenizer, lang)
    return DataLoader(dataset, batch_size=64, shuffle=shuffle, collate_fn=collate_fn)