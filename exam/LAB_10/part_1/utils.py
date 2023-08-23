import json
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os

PAD_TOKEN = 0

def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def get_raw_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'ATIS')
    X_train = load_data(os.path.join(dataset_path, 'train.json'))
    X_test = load_data(os.path.join(dataset_path, 'test.json'))

    # Calculate the portion of data to use for dev set
    total_data_len = len(X_train) + len(X_test)
    portion = round(total_data_len * 0.10 / len(X_train), 2)

    # Stratify based on intents
    intents = [x['intent'] for x in X_train]
    intent_counts = Counter(intents)

    # Split the training data into two lists: 
    # 1) those with intents appearing more than once
    # 2) those with intents appearing only once
    multi_intent_data = [X_train[i] for i, intent in enumerate(intents) if intent_counts[intent] > 1]
    single_intent_data = [X_train[i] for i, intent in enumerate(intents) if intent_counts[intent] == 1]

    # Perform stratified train-dev split
    multi_intents = [x['intent'] for x in multi_intent_data]
    X_train, X_dev = train_test_split(multi_intent_data, test_size=portion, random_state=42, stratify=multi_intents)

    # Add the single occurrence intent data to the training set
    X_train.extend(single_intent_data)

    return X_train, X_dev, X_test

def get_lang(train_raw, dev_raw, test_raw):
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    return Lang(words, intents, slots, cutoff=0)

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
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
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.LongTensor(self.utt_ids[idx])
        slots = torch.LongTensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
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
    
def collate_fn(batch):
    def pad_to_max_len(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, lengths
    # Sort data by sequence lengths
    batch.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {key: [d[key] for d in batch] for key in batch[0].keys()}
    
    src_utt, _ = pad_to_max_len(new_item['utterance'])
    y_slots, y_lengths = pad_to_max_len(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    batch = {
        "utterances": src_utt,
        "intents": intent,
        "y_slots": y_slots,
        "slots_len": torch.LongTensor(y_lengths)
    }
    return batch

def get_dataloaders(train_raw, dev_raw, test_raw, lang):
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader