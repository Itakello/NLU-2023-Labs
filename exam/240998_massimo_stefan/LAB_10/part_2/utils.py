import json
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
from transformers import BertTokenizer
import torch.nn.functional as F

PAD_TOKEN = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True) # unk required for OOV
        self.slot2id = self.lab2id(slots, pad=True)
        self.intent2id = self.lab2id(intents, pad=False) # not pad for intents
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=0, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab
    
class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.tokenizer = tokenizer
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        # Tokenize the utterance with the BERT tokenizer, also truncate phrases longer than 512 tokens
        tokens = self.tokenizer(self.utterances[idx], return_tensors="pt", padding=False, truncation=True)
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        slots = self.get_slots(idx, input_ids)        
        intent = self.intent_ids[idx]
        sample = {'input_ids': input_ids, 'attention_mask': attention_mask, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def get_slots(self, idx, input_ids): # handle subword tokens vs slot labels at word level
        slot_labels = self.slot_ids[idx]
        aligned_slot_labels = []
        word_idx = 0
        for token in input_ids:
            # if [CLS] and [SEP] tokens
            if token == self.tokenizer.sep_token_id or token == self.tokenizer.cls_token_id:
                aligned_slot_labels.append(PAD_TOKEN)  # Ignore them
            # if [UNK] token
            elif token == self.tokenizer.unk_token_id:
                aligned_slot_labels.append(slot_labels[word_idx])
                word_idx += 1
            # if token is a full word 
            elif self.tokenizer.decode([token]) == self.utterances[idx].split()[word_idx]:
                aligned_slot_labels.append(slot_labels[word_idx])
                word_idx += 1
            # if token is a subword, all the tokens in the word get the same label
            else:
                aligned_slot_labels.append(slot_labels[word_idx])

        return torch.Tensor(aligned_slot_labels)
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['input_ids']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_input_ids, _ = merge(new_item['input_ids'])
    src_attention_mask, _ = merge(new_item['attention_mask'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_input_ids = src_input_ids.to(device)
    src_attention_mask = src_attention_mask.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    # Pad y_slots to match the shape of the output tensor
    max_len = src_input_ids.size(1)
    if y_slots.size(1) < max_len:
        y_slots = F.pad(y_slots, (0, max_len - y_slots.size(1)))
    
    new_item["input_ids"] = src_input_ids
    new_item["attention_mask"] = src_attention_mask
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


def get_raw_data():
    tmp_train_raw = load_data(os.path.join(os.path.dirname(__file__),'ATIS','train.json'))
    test_raw = load_data(os.path.join(os.path.dirname(__file__),'ATIS','test.json'))
    
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10)/(len(tmp_train_raw)),2)

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occure once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw, test_raw

def get_dataloaders():
    train_raw, dev_raw, test_raw = get_raw_data()
    lang = get_lang()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader

def get_lang():
    train_raw, dev_raw, test_raw = get_raw_data()
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    return Lang(words, intents, slots, cutoff=0)