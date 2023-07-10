import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropout_rate=0.5):
        super(ModelIAS, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # ! Make LSTM bidirectional
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True)    
        self.slot_out = nn.Linear(hid_size * 2, out_slot)  # ! Adjust input size
        self.intent_out = nn.Linear(hid_size * 2, out_int)  # ! Adjust input size
        
        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = utt_emb.permute(1,0,2) # we need seq len first -> seq_len X batch_size X emb_size
        
        # avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        # ! Apply dropout
        utt_encoded = self.dropout(utt_encoded)
        # ! Concatenate the last hidden states of the forward and backward LSTM layers
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        # Get the last hidden state
        last_hidden = self.dropout(last_hidden)
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        # Slot size: seq_len, batch size, calsses 
        slots = slots.permute(1,2,0) # required to compute the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent