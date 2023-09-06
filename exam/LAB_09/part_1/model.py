import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_id, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output

class LM_LSTM_Dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_id, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.out_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output