import torch.nn as nn

# Use same droput mask across time-steps
class VariationalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x

        if self.mask is None or self.mask.size(1) != x.size(1) or self.mask.size(2) != x.size(2):
            self.mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)

        return x * self.mask.div_(1 - self.p)


class LM_LSTM_Adv(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_id=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_Adv, self).__init__()
        assert emb_size == hidden_size, "emb_size must be equal to hidden_size for weight tying"
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_id)
        self.emb_dropout = VariationalDropout(emb_dropout) #! Variational droupout
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.out_dropout = VariationalDropout(out_dropout) #! Variational droupout
        self.pad_token = pad_id
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight  # ! Weight tying
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output