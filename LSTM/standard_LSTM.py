import torch
import torch.autograd as autograd # torch中自動計算梯度模塊
import torch.nn as nn             # 神經網絡模塊
import torch.nn.functional as F   # 神經網絡模塊中的常用功能 

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

is_cuda = torch.cuda.is_available()

class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, embedding_weights,
                 hidden_dim, tag_to_ix, dropout, num_layers, bidirectional):
        
        super(LSTMTagger, self).__init__()
        
        self.direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim // self.direction
        self.num_layers = num_layers
        self.tagset_size = len(tag_to_ix)

        weights = torch.cuda.FloatTensor(embedding_weights) if is_cuda else torch.FloatTensor(embedding_weights)
        self.word_embeddings = nn.Embedding.from_pretrained(weights, freeze=True)
        
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
                            dropout=dropout, num_layers=self.num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
 
        self.hidden2tag = nn.Linear(self.hidden_dim * self.direction, self.tagset_size)
 

    def init_hidden(self, batch_size):
        h_states = autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim))
        c_states = autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim))
        
        return (h_states.cuda(), c_states.cuda()) if is_cuda else (h_states, c_states)

        
    def forward(self, sentence, lengths):
        batch_size, seq_len = sentence.shape
        self.hidden = self.init_hidden(batch_size)
        
        try:
            embeds = self.word_embeddings(sentence) # [batch_size, seq_len, emb_dim]
            embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)
            lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True)

            tag_space = self.hidden2tag(lstm_out.contiguous().view(batch_size * seq_len, -1))
            tag_scores = F.log_softmax(tag_space, dim=1)
    
            return tag_scores
        
        except Exception as e:
            print(sentence.shape)
            print(embeds.shape)
            print(e)