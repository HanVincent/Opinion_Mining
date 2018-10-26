
# coding: utf-8

# In[21]:

from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.autograd as autograd # torch中自動計算梯度模塊
import torch.nn as nn             # 神經網絡模塊
import torch.nn.functional as F   # 神經網絡模塊中的常用功能 
import torch.optim as optim       # 模型優化器模塊
import numpy as np
import pickle
import math, datetime, time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["LANG"] = "en_US.UTF-8"
# os.environ["LC_CTYPE"] = "en_US.UTF-8"

is_cuda = torch.cuda.is_available()


# In[22]:

def get_sentence_target(entry):
    sentence, target = [], []
    for line in entry.split('\n'):
        if line.strip() == '': continue
                
        token, pos, bio = line.split('\t')
    
        if token in ['-LRB-', '-LSB-', '-LCB-']: token = '('
        elif token in ['-RRB-', '-RSB-', '-RCB-']: token = ')'
            
        sentence.append(token)
        target.append(bio)
        
    return sentence, target


def group_data(file):
    sentenceIDs = open('dataset/sentenceid.txt', 'r', encoding='utf8').read().strip().split('\n')
    entries = open(file, 'r', encoding='utf8').read().strip().split('\n\n')

    assert len(sentenceIDs) == len(entries)
    
    documents = defaultdict(lambda: [])
    for sent_id, entry in zip(sentenceIDs, entries):
        sent_id = sent_id.split(' ')[2]
        sentence, target = get_sentence_target(entry)        
        documents[sent_id].append((sentence, target))

    return documents


# In[23]:

def split_dataset(documents, num):
    docs_list = open('dataset/datasplit/doclist.mpqaOriginalSubset', 'r', encoding='utf8').read().strip().split('\n')
    train_ids = open(f'dataset/datasplit/filelist_train{num}', 'r', encoding='utf8').read().strip().split('\n')
    test_ids = open(f'dataset/datasplit/filelist_test{num}', 'r', encoding='utf8').read().strip().split('\n')
    
    train, test, dev = [], [], []
    for doc_id in docs_list:
        if   doc_id in train_ids: train.extend(documents[doc_id])
        elif doc_id in test_ids: test.extend(documents[doc_id])
        else: dev.extend(documents[doc_id])
    
    train.sort(key=lambda pair: len(pair[1]), reverse=True)
    test.sort(key=lambda pair: len(pair[1]), reverse=True)
    dev.sort(key=lambda pair: len(pair[1]), reverse=True)
    
    print("Train size: {}, Test size: {}, Dev size: {}".format(len(train), len(test), len(dev)))
    
    return train, test, dev


# In[24]:

def sequence_to_ixs(seq, to_ix):
    ixs = [to_ix[w] if w in to_ix else to_ix[UNK_TOKEN] for w in seq]
    return torch.cuda.LongTensor(ixs) if is_cuda else torch.LongTensor(ixs)


def ixs_to_sequence(seq, to_word):
    tokens = [to_word[ix] for ix in seq]
    return tokens


def padding(seq, max_size):
    diff = max_size - len(seq)
    return seq + [PAD_TOKEN] * diff


def batch_padding(seqs, max_size):
    return [ padding(seq, max_size) for seq in seqs]


def batch_seq_to_idx(seqs, to_ix):
    return [ sequence_to_ixs(seq, to_ix) for seq in seqs]


# In[140]:

class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, 
                 tagset_size, 
                 dropout, num_layers, bidirectional):
        super(LSTMTagger, self).__init__()
        
        self.direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim // self.direction
        self.num_layers = num_layers
        self.tagset_size = tagset_size

        weights = torch.cuda.FloatTensor(embedding_weights) if is_cuda else torch.FloatTensor(embedding_weights)
        self.word_embeddings = nn.Embedding.from_pretrained(weights, freeze=True)
        
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
                            dropout=dropout, num_layers=self.num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
 
        self.hidden2tag = nn.Linear(self.hidden_dim * self.direction, self.tagset_size)
 

    def init_hidden(self, batch_size):
        if is_cuda:
            return (autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim).cuda()))
        else:
            return (autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_dim)))

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


# In[26]:

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

            lengths = list(map(lambda x: x.shape[0], x))

            padded_seqs = pad_sequence(x, batch_first=True)
            padded_tags = pad_sequence(y, batch_first=True)

            predict_tags = model(padded_seqs, lengths)
            true_tags = padded_tags.view(-1)

            loss = loss_function(predict_tags, true_tags)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print("epoch: {}, loss: {}".format(epoch+1, loss))
            # torch.save(model.state_dict(), model_path)


# In[32]:

from evaluate import *

def test(test_data):
    with torch.no_grad():
        data = test_data
        
        x = list(map(lambda pair: sequence_to_ixs(pair[0], word_to_ix), data))
        y = list(map(lambda pair: sequence_to_ixs(pair[1], tag_to_ix), data))

        lengths = list(map(lambda x: x.shape[0], x))

        padded_seqs = pad_sequence(x, batch_first=True)
        y_predicts = model(padded_seqs, lengths)
        y_predicts = torch.max(y_predicts, 1)[1].view([len(lengths), -1])

        y_trues = y
        y_predicts = [y_[:lengths[i]] for i, y_ in enumerate(y_predicts)]

        # 感覺可以實驗 tag by tag
        result = evaluate(y_predicts, y_trues)
        
        return result, (y_predicts, y_trues)


# In[136]:

# Constant
UNK_TOKEN = '<UNK>'

# Data 
file_name = 'dataset/dse.txt'

# Store model
model_path = 'models/' + datetime.datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d_%H%M") + '.model'

# Word embeddings
source = 'glove'

# Model hyper-parameters
embedding_dim = 300
hidden_dim = 100
learning_rate = 0.01
momentum = 0.7
dropout = 0
num_layers = 3
bidirectional = True
batch_size = 80
epochs = 200


# In[79]:

### Get Word Embeddings
with open(f'dataset/{source}.pickle', 'rb') as handle:
    word_vectors, embedding_weights, word_to_ix, ix_to_word = pickle.load(handle)

### Manual Tag
tag_to_ix = {"B": 0, "I": 1, "O": 2}
ix_to_tag = {0: "B", 1: "I", 2: "O"}


# In[81]:

best_result = 0
results = []
for num in range(10):
    print("10-fold:", num, "="*50)
    
    # Get Data and split
    documents = group_data(file_name)
    train_data, test_data, dev_data = split_dataset(documents, num)

    # Create Model
    model = LSTMTagger(embedding_dim, hidden_dim, 
                       len(tag_to_ix), 
                       dropout=dropout,
                       num_layers=num_layers,
                       bidirectional=bidirectional)

    loss_function = nn.NLLLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)

    # If GPU available, use GPU 
    if is_cuda: 
        model.cuda()
        
    train(train_data)
    
    result, _ = test(test_data)
    
    if result['proportional']['f1'] >= best_result:
        best_result = result['proportional']['f1']        
        torch.save(model.state_dict(), model_path)
        print("Store Model with score: {}".format(best_result))
        
    results.append(result)


# In[143]:

bin_result = { 'precision': .0, 'recall': .0, 'f1': .0 }
prop_result = { 'precision': .0, 'recall': .0, 'f1': .0 }

for i, result in enumerate(results):
    for key in result['binary']: bin_result[key] += (result['binary'][key] / len(results))
    for key in result['proportional']: prop_result[key] += (result['proportional'][key] / len(results))
    
    print("10-fold: {}".format(i))
    print("Binary Overlap\t\tPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}".format(**result['binary']))
    print("Proportional Overlap\tPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}".format(**result['proportional']))

print("\nAverage", "="*70)
print("Binary Overlap\t\tPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}".format(**bin_result))
print("Proportional Overlap\tPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}".format(**prop_result))


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




# In[ ]:




# In[141]:

# # Get Data and split
# documents = group_data(file_name)
# train_data, test_data, dev_data = split_dataset(documents, 0)

# # Create Model
# model = LSTMTagger(embedding_dim, hidden_dim, 
#                    len(tag_to_ix), 
#                    dropout=dropout,
#                    num_layers=num_layers,
#                    bidirectional=bidirectional)

# # model.load_state_dict(torch.load(model_path))

# # If GPU available, use GPU 
# if is_cuda: 
#     model.cuda()

# result, y_pair = test(test_data)
# print(result)


# In[142]:

# for name, param in model.named_parameters():
#     print( name, param.shape)
    
# total_param = sum(p.numel() for p in model.parameters())
# print(total_param)


# In[ ]:



