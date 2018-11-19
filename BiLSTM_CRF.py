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


class BiLSTM_CRF(nn.Module):
 
    def __init__(self, embedding_dim, embedding_weights,
                 hidden_dim, tag_to_ix, dropout, num_layers, bidirectional):
        
        super(BiLSTM_CRF, self).__init__()
        
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
 

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim * self.direction, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        init_transitions = torch.randn(self.tagset_size, self.tagset_size)
   
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        init_transitions.data[tag_to_ix[START_TAG], :] = -10000.0
        init_transitions.data[:, tag_to_ix[STOP_TAG]] = -10000.0
        
        if is_cuda: init_transitions = init_transitions.cuda()

        self.transitions = nn.Parameter(init_transitions)
        
        self = self.cuda() if is_cuda else self
        
      
    def init_hidden(self, batch_size):
#         h_states = autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim))
#         c_states = autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim))

        h_states = torch.randn(self.num_layers * self.direction, batch_size, self.hidden_dim)
        c_states = torch.randn(self.num_layers * self.direction, batch_size, self.hidden_dim)
        
        return (h_states.cuda(), c_states.cuda()) if is_cuda else (h_states, c_states)


    def _forward_alg(self, feats, mask):
        batch_size, seq_len, tagset_size = feats.shape
        
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.) # [B, C]
        if is_cuda: init_alphas = init_alphas.cuda()
        
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        
        trans = self.transitions.unsqueeze(0) # [1, C, C]

        # Wrap in a variable so that we will get automatic backprop
        score = init_alphas # forward_var
        
        # Iterate through the sentence
        for t in range(seq_len): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = feats[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
            
        score = log_sum_exp(score + self.transitions[tag_to_ix[STOP_TAG]]) # termination
        
        # return alpha
        return score # partition function
    

    def _get_lstm_features(self, sentences, mask):
        batch_size, seq_len = sentences.shape
        
        self.hidden = self.init_hidden(batch_size)

        embeds = self.word_embeddings(sentences)
        embeds = pack_padded_sequence(embeds, mask.sum(1).int(), batch_first=True)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True)
        
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # lstm_out = lstm_out.contiguous().view(batch_size * seq_len, -1)
        
        lstm_feats = self.hidden2tag(lstm_out) 
        
        lstm_feats = lstm_feats * mask.unsqueeze(-1)
        # lstm_feats = lstm_feats.view(batch_size, seq_len, -1)
        
        return lstm_feats

    
    # calculate the score of a given sequence 
    def _score_sentence(self, feats, tags, mask):
        batch_size, seq_len, tagset_size = feats.shape
        
        score = torch.zeros(batch_size)
        if is_cuda: score = score.cuda()
        
        feats = feats.unsqueeze(3)
        trans = self.transitions.unsqueeze(2)
        
        start_pad = torch.cuda.LongTensor( batch_size, 1 ).fill_(tag_to_ix[START_TAG])
        tags = torch.cat([start_pad, tags], dim=1)
        
        for t in range(seq_len):
            mask_t = mask[:, t]
            emit_t = torch.cat([feats[b, t, tags[b][t + 1]] for b in range(batch_size)])
            trans_t = torch.cat([trans[seq[t + 1], seq[t]] for seq in tags])
            score += (emit_t + trans_t) * mask_t
    
        return score

    
    # initialize backpointers and viterbi variables in log space
    def _viterbi_decode(self, feats, mask):    
        batch_size, seq_len, tagset_size = feats.shape
        
        if is_cuda:
            bptr = torch.LongTensor().cuda()
            score = torch.full((batch_size, self.tagset_size), -10000.).cuda()
        else:
            bptr = torch.LongTensor()
            score = torch.full((batch_size, self.tagset_size), -10000.)
                
        score[:, tag_to_ix[START_TAG]] = 0.

        for t in range(seq_len): # recursion through the sequence
            # backpointers and viterbi variables at this timestep
            if is_cuda:
                bptr_t = torch.LongTensor().cuda()
                score_t = torch.Tensor().cuda()
            else:
                bptr_t = torch.LongTensor()
                score_t = torch.Tensor()
            
            # TODO: vectorize
            for i in range(self.tagset_size): # for each next tag
                m = [j.unsqueeze(1) for j in torch.max(score + self.transitions[i], 1)]
                bptr_t  = torch.cat((bptr_t, m[1]), 1)  # best previous tags
                score_t = torch.cat((score_t, m[0]), 1) # best transition scores
            
            if is_cuda:
                bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            else:
                bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t + feats[:, t] # plus emission scores
            
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        # TODO: must cpu list?
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        
        for b in range(batch_size):
            x = best_tag[b] # best tag
            l = mask[b].sum().int().tolist()
            for bptr_t in reversed(bptr[b][:l]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        # return best_path
        return best_score, best_path


    def neg_log_likelihood(self, sentences, true_tags, mask):
        
        feats = self._get_lstm_features(sentences, mask)
        
        forward_score = self._forward_alg(feats, mask)
        
        gold_score = self._score_sentence(feats, true_tags, mask)
        
        return forward_score - gold_score

    
    def forward(self, sentence, mask):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, mask)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats, mask)
        
        return score, tag_seq


# In[ ]:


def sequence_to_ixs(seq, to_ix):
    ixs = [to_ix[w] if w in to_ix else to_ix[UNK_TOKEN] for w in seq]
    return torch.cuda.LongTensor(ixs) if is_cuda else torch.LongTensor(ixs)


def ixs_to_sequence(seq, to_word):
    tokens = [to_word[ix] for ix in seq]
    return tokens


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))


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

            # lengths = list(map(lambda x: x.shape[0], x))

            padded_seqs = pad_sequence(x, batch_first=True)
            padded_tags = pad_sequence(y, batch_first=True)

            mask = padded_tags.data.gt(0).float()

            true_tags = padded_tags

            loss_function = model.neg_log_likelihood(padded_seqs, true_tags, mask)
            # predict_tags = model(padded_seqs, lengths)
            # loss = loss_function(predict_tags, true_tags)
            # loss = model.loss(predict_tags, true_tags)
            
            optimizer.zero_grad()
            
            loss = torch.mean(loss_function)
            
            loss.backward()
            
            optimizer.step()
            

        if (epoch + 1) % 5 == 0:
            print("epoch: {}, loss: {}".format(epoch+1, loss))
            
            # writer.add_scalar('Train/Loss'.format(epoch), loss.data[0], epoch)


# In[ ]:


from utils.evaluate import evaluate
from utils.constant import *


# In[ ]:


def test(test_data):
    with torch.no_grad():
        data = test_data
        
        x = list(map(lambda pair: sequence_to_ixs(pair[0], word_to_ix), data))
        y = list(map(lambda pair: sequence_to_ixs(pair[1], tag_to_ix), data))

        padded_seqs = pad_sequence(x, batch_first=True)
        padded_tags = pad_sequence(y, batch_first=True)

        mask = padded_tags.data.gt(0).float() # PAD = 0
        
        score, y_predicts = model(padded_seqs, mask) 
        
        # y_predicts = torch.max(y_predicts, 2)[1].view([len(x), -1])
        
        y_trues = y

        # y_predicts = [y_[:len(y_trues[i])] for i, y_ in enumerate(y_predicts)]

        result = evaluate(y_predicts, y_trues)

        return result, (y_predicts, y_trues)


# In[ ]:


# Data 
file_name = 'dataset/dse.txt'

# Store model
model_path = 'models/' + datetime.datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d_%H%M") + '.model'

# Word embeddings
source = 'word2vec'

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


# import sys

# argv = sys.argv[1:]

# source = argv[0]
# hidden_dim = int(argv[1])
# learning_rate = float(argv[2])
# num_layers = int(argv[3])
# bidirectional = int(argv[4])
# dropout = float(argv[5])
# batch_size = int(argv[6])


# In[ ]:


### Get Word Embeddings
with open(f'dataset/{source}.pickle', 'rb') as handle:
    word_vectors, embedding_weights, word_to_ix, ix_to_word = pickle.load(handle)


# In[ ]:


# model = BiLSTM_CRF(embedding_dim, embedding_weights,
#                    hidden_dim, tag_to_ix, 
#                    dropout=dropout,num_layers=num_layers,
#                    bidirectional=bidirectional)

# if is_cuda: model.cuda()

# train_data = [(
#     "the wall street journal reported today that apple corporation made money".split(),
#     "B I I I O O O B I O O".split()
# ), (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )]

# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
#                           lr=learning_rate, momentum=momentum)

# train(train_data)

# test(train_data)


# In[ ]:


best_result = 0
results = []
for num in range(10):
    print("10-fold:", num, "="*50)
    
    # Get Data and split
    documents = group_data(file_name)
    train_data, test_data, dev_data = split_dataset(documents, num)

    # Create Model
    model = BiLSTM_CRF(embedding_dim, embedding_weights,
                       hidden_dim, tag_to_ix, 
                       dropout=dropout,num_layers=num_layers,
                       bidirectional=bidirectional)

    if is_cuda: model.cuda()
        
    # loss_function = nn.NLLLoss()

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


# model_path = 'models/20181110_1453.model'
# fname = 'dse'

# # Get Data and split
# documents = group_data(file_name)
# train_data, test_data, dev_data = split_dataset(documents, 5)


# # Create Model
# model = BiLSTM_CRF(embedding_dim, embedding_weights,
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




