# import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    
    def __init__(self, embedder, input_size, hidden_size, tagset_size, 
                 num_layers, bidirectional, dropout):
        super().__init__()

        self.embedder = embedder
        self.input_size = input_size
        self.direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.direction
        self.tagset_size = tagset_size,
        self.num_layers = num_layers,
        
        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = True,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout
        )
        self.out = nn.Linear(self.hidden_size, self.tagset_size) # LSTM output to tag

        
    def init_hidden(self, batch_size): # initialize hidden states
        h = zeros(self.num_layers * self.direction, batch_size, self.hidden_size) # hidden states
        c = zeros(self.num_layers * self.direction, batch_size, self.hidden_size) # cell states
        
        return (h, c)

    def forward(self, x, mask):
        batch_size, seq_len = x.shape
        
        self.hidden = self.init_hidden(batch_size)
        
        x = self.embedder(x)
        
        x = pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.lstm(x, self.hidden)
        h, _ = pad_packed_sequence(h, batch_first = True)
        
        h = self.out(h)
        
        h *= mask.unsqueeze(2)
        
        return h