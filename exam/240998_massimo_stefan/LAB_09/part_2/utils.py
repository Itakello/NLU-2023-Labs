# Add functions or classes used for data loading and preprocessing
import os
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader

def read_file(path, eos_token="<eos>"): 
    output = []
    file_path = os.path.join(os.path.dirname(__file__), path)
    with open(file_path, "r") as f:
        for line in f.readlines():
            cleaned_line = line.strip()
            output.append(cleaned_line + " " + eos_token)
    return output

def get_raw_data(): 
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")
    return train_raw, dev_raw, test_raw

class Lang(): 
    def __init__(self, corpus, special_tokens):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    
class PennTreeBank (data.Dataset): 
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
     
    def mapping_seq(self, data, lang): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                new_word = "<unk>"
                if x in lang.word2id:
                    new_word = x
                tmp_seq.append(lang.word2id[new_word])
            res.append(tmp_seq)
        return res
    
def extend_to_max_len(sequences, pad_token): 
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

def merge_data(data): 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    return new_item
  
def collate_fn(data, pad_token): 
    data.sort(key=lambda x: len(x["source"]), reverse=True) # Sort by decreasing order of length, beneficial for RNNs
    
    # Merge data for batch
    batch = merge_data(data)

    batch["source"], _ = extend_to_max_len(batch["source"], pad_token)
    batch["target"], lengths = extend_to_max_len(batch["target"], pad_token)
    batch["number_tokens"] = sum(lengths)
    return batch
    
def get_dataloaders(train_raw, dev_raw, test_raw, lang, pad_id, batch_size=256): 
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    new_collate_fn = partial(collate_fn, pad_token=pad_id)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=new_collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=new_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=new_collate_fn)
    
    return train_loader, dev_loader, test_loader