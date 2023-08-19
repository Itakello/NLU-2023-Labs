import json
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import os
from transformers import BertTokenizer
import torch.nn.functional as F

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

"""def get_lang(train_raw, dev_raw, test_raw):
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    return Lang(words, intents, slots, cutoff=0)"""

def get_label_vocab(corpus, key):
    labels = set([item[key] for item in corpus])
    return {label: i for i, label in enumerate(labels)}

def get_vocab(train_raw, dev_raw, test_raw):
    corpus = train_raw + dev_raw + test_raw
    
    # Build slot and intent vocabularies
    slot_vocab = get_label_vocab(corpus, 'slots')
    intent_vocab = get_label_vocab(corpus, 'intent')
    
    return slot_vocab, intent_vocab

class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = [x['utterance'] for x in dataset]
        self.unk = unk
        self.slot_ids = self.mapping_seq([x['slots'] for x in dataset], lang.slot2id)
        self.intent_ids = self.mapping_lab([x['intent'] for x in dataset], lang.intent2id)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        tokens, aligned_slot_labels = self.align_slot_labels(self.utterances[idx], self.slot_ids[idx])
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        attention_mask = [1] * len(input_ids)
        intent = self.intent_ids[idx]
        
        sample = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'slots': torch.tensor(aligned_slot_labels),
            'intent': intent,
            'slots_len': len(aligned_slot_labels)
        }
        return sample
    
    def align_slot_labels(self, sentence, slot_labels):
        tokens = []
        aligned_labels = []
        
        # Tokenize each word separately to ensure alignment with original words
        for word, label in zip(sentence.split(), slot_labels):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Repeat the slot label for each sub-token of the word
            aligned_labels.extend([label] * len(word_tokens))

        return tokens, aligned_labels

    def mapping_lab(self, data, mapper):
        return [mapper.get(x, mapper[self.unk]) for x in data]

    def mapping_seq(self, data, mapper):
        return [[mapper.get(x, mapper[self.unk]) for x in seq.split()] for seq in data]
    
def collate_fn(data):
    def pad_to_max_len(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs

    # Sort data by sequence lengths for efficient RNN packing, if necessary
    data.sort(key=lambda x: len(x['input_ids']), reverse=True) 
    batched_item = {}
    for key in data[0].keys():
        batched_item[key] = [d[key] for d in data]

    batched_item["input_ids"] = pad_to_max_len(batched_item['input_ids'])
    batched_item["attention_mask"] = pad_to_max_len(batched_item['attention_mask'])
    batched_item["slots"] = pad_to_max_len(batched_item["slots"])
    batched_item["intent"] = torch.LongTensor(batched_item["intent"])
    batched_item["slots_len"] = torch.LongTensor([d["slots_len"] for d in data])

    return batched_item

def get_dataloaders(train_raw, dev_raw, test_raw, lang):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = IntentsAndSlots(train_raw, lang, tokenizer)
    dev_dataset = IntentsAndSlots(dev_raw, lang, tokenizer)
    test_dataset = IntentsAndSlots(test_raw, lang, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader