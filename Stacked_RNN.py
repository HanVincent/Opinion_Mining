#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.optim as optim       # 模型優化器模塊
import torch.autograd as autograd # torch中自動計算梯度模塊
import torch.nn.functional as F   # 神經網絡模塊中的常用功能 

import numpy as np
import pickle, math, datetime, time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

is_cuda = torch.cuda.is_available()


# In[ ]:


# from tensorboardX import SummaryWriter
# writer = SummaryWriter('log')


# In[ ]:


from utils.preprocess import get_sentence_target, group_data, split_dataset


# In[ ]:


class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, embedding_weights,
                 hidden_dim, tag_to_ix, dropout, num_layers, bidirectional):
        
        super(LSTMTagger, self).__init__()
        
        self.direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim // self.direction
        self.num_layers = num_layers
        self.tag_to_ix = tag_to_ix
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

        
    def _forward_alg(self, sentences, mask):
        batch_size, seq_len = sentences.shape
        
        self.hidden = self.init_hidden(batch_size)

        embeds = self.word_embeddings(sentences) # [batch_size, seq_len, emb_dim]
        embeds = pack_padded_sequence(embeds, mask.sum(1).int(), batch_first=True)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True)

        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2)

        y_ = torch.mul(tag_scores, mask.unsqueeze(-1).expand([batch_size, seq_len, self.tagset_size]))
        
        return y_

        
    def forward(self, sentences, mask):
        y_ = self._forward_alg(sentences, mask)

        return y_
            
            
    def loss(self, sentences, y, mask):
        batch_size, seq_len = sentences.shape
        
        y_ = self._forward_alg(sentences, mask)
        y_ = y_.view(batch_size*seq_len, -1)
        y  = y.view(-1)
        
        loss_ = loss_function(y_, y)

        loss_.backward()
        
        return loss_


# In[ ]:


def sequence_to_ixs(seq, to_ix):
    ixs = [to_ix[w] if w in to_ix else to_ix[UNK_TOKEN] for w in seq]
    
    return torch.cuda.LongTensor(ixs) if is_cuda else torch.LongTensor(ixs)


def ixs_to_sequence(seq, to_word):
    tokens = [to_word[ix] for ix in seq]
    
    return tokens


# In[ ]:


def train(training_data):
    total_num = len(training_data)
    batch_num = math.ceil(total_num / batch_size)

    for epoch in range(epochs):
        
        for i in range(batch_num):
            model.zero_grad()

            data = training_data[i * batch_size : (i+1) * batch_size]

            x = list(map(lambda pair: sequence_to_ixs(pair[0], word_to_ix), data))
            y = list(map(lambda pair: sequence_to_ixs(pair[1], tag_to_ix), data))

            assert len(x) == len(y)
            
            padded_seqs = pad_sequence(x, batch_first=True)
            padded_tags = pad_sequence(y, batch_first=True)
        
            mask = padded_tags.data.gt(0).float() # PAD = 0
            
            true_tags = padded_tags

            optimizer.zero_grad()
            
            loss = model.loss(padded_seqs, true_tags, mask)
            
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print("epoch: {}, loss: {}".format(epoch+1, loss))
            
            # writer.add_scalar('Train/Loss'.format(epoch), loss.data[0], epoch)


# In[ ]:


from utils.evaluate import evaluate
from utils.constant import *

def test(test_data):
    with torch.no_grad():
        data = test_data
        
        x = list(map(lambda pair: sequence_to_ixs(pair[0], word_to_ix), data))
        y = list(map(lambda pair: sequence_to_ixs(pair[1], tag_to_ix), data))

        padded_seqs = pad_sequence(x, batch_first=True)
        padded_tags = pad_sequence(y, batch_first=True)

        mask = padded_tags.data.gt(0).float() # PAD = 0
        
        y_predicts = model(padded_seqs, mask) #[80, 169, 4]
        
        y_predicts = torch.max(y_predicts, 2)[1].view([len(x), -1])
        
        y_trues = y
        
        y_predicts = [y_[:len(y_trues[i])] for i, y_ in enumerate(y_predicts)]

        result = evaluate(y_predicts, y_trues)

        return result, (y_predicts, y_trues)


# In[ ]:


# Data 
file_name = 'dataset/dse.txt'

# Store model
model_path = 'models/' + datetime.datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d_%H%M") + '.model'

# Word embeddings
source = 'glove'

# Model hyper-parameters
embedding_dim = 300
hidden_dim = 100
learning_rate = 0.03
momentum = 0.7
num_layers = 3
bidirectional = True
dropout = 0
batch_size = 80
epochs = 200


# In[ ]:


import sys

argv = sys.argv[1:]

source = argv[0]
hidden_dim = int(argv[1])
learning_rate = float(argv[2])
num_layers = int(argv[3])
bidirectional = int(argv[4])
dropout = float(argv[5])
batch_size = int(argv[6])


# In[ ]:


### Get Word Embeddings
with open(f'dataset/{source}.pickle', 'rb') as handle:
    word_vectors, embedding_weights, word_to_ix, ix_to_word = pickle.load(handle)


# In[ ]:


best_result = 0
results = []
for num in range(10):
    print("10-fold:", num, "="*50)
    
    # Get Data and split
    documents = group_data(file_name)
    train_data, test_data, dev_data = split_dataset(documents, num)

    # Create Model
    model = LSTMTagger(embedding_dim, embedding_weights,
                       hidden_dim, tag_to_ix, 
                       dropout=dropout,num_layers=num_layers,
                       bidirectional=bidirectional)

    if is_cuda: model.cuda()
        
    loss_function = nn.NLLLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate, momentum=momentum)

    train(train_data)

    result, _ = test(test_data)
    
    if result['proportional']['f1'] >= best_result:
        best_result = result['proportional']['f1']        
        torch.save(model.state_dict(), model_path)
        print("Store Model with score: {}".format(best_result))
        
    results.append(result)


# In[ ]:


bin_result = { 'precision': .0, 'recall': .0, 'f1': .0 }
prop_result = { 'precision': .0, 'recall': .0, 'f1': .0 }

for i, result in enumerate(results):
    for key in result['binary']: bin_result[key] += (result['binary'][key] / len(results))
    for key in result['proportional']: prop_result[key] += (result['proportional'][key] / len(results))
    
    print("10-fold: {}".format(i))
    print("Binary Overlap\t\tPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}".format(**result['binary']))
    print("Proportional Overlap\tPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}".format(**result['proportional']))

print("\nAverage", "="*70)
print("Binary Overlap\t\tPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}".format(**bin_result))
print("Proportional Overlap\tPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}".format(**prop_result))


print("\nParams", "=" * 70)
print(f'''model_path = {model_path}
file_name = {file_name}
source = {source}
embedding_dim = {embedding_dim}
hidden_dim = {hidden_dim}
learning_rate = {learning_rate}
momentum = {momentum}
dropout = {dropout}
num_layers = {num_layers}
bidirectional = {bidirectional}
batch_size = {batch_size}
epochs = {epochs}''')


# In[ ]:





# ### Load model and observe the prediction

# In[ ]:


# model_path = 'models/20181105_1001.model'
# fname = 'dse'

# # Get Data and split
# documents = group_data(file_name)
# train_data, test_data, dev_data = split_dataset(documents, 5)


# # Create Model
# model = LSTMTagger(embedding_dim, embedding_weights,
#                    hidden_dim, tag_to_ix,
#                    dropout=dropout,
#                    num_layers=num_layers,
#                    bidirectional=bidirectional)

# model.load_state_dict(torch.load(model_path))

# if is_cuda: model.cuda()

# result, y_pair = test(test_data)

# print(result)


# In[ ]:


# ys_, ys = y_pair

# ws = open(f'dataset/failure_{fname}.txt', 'w', encoding='utf8')
# correct = 0
# for (tks, tags), y_, y in zip(test_data, ys_, ys):
#     if sum(torch.eq(y_, y)) == len(tks):
#         correct += 1
#     else:
#         sents, trues, bios = [], [], []
#         for i, tk in enumerate(tks):
#             length = len(tk)
#             sents.append(tk)
#             bios.append('{:>{length}s}'.format(ix_to_tag[int(y_[i])], length=length))
#             trues.append('{:>{length}s}'.format(ix_to_tag[int(y[i])], length=length))
            
#         print(' '.join(sents), file=ws)
#         print(' '.join(bios), file=ws)
#         print(' '.join(trues), file=ws)
#         print("="*20, file=ws)
        
# ws.close()
# print(correct / len(test_data))


# ### Calculate the number of parameters

# In[ ]:


# for name, param in model.named_parameters():
#     print( name, param.shape)
    
# total_param = sum(p.numel() for p in model.parameters())
# print(total_param)


# In[ ]:




