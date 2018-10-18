
# coding: utf-8

# In[1]:

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.autograd as autograd # torch中自動計算梯度模塊
import torch.nn as nn             # 神經網絡模塊
import torch.nn.functional as F   # 神經網絡模塊中的常用功能 
import torch.optim as optim       # 模型優化器模塊
import numpy as np
import math
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_CTYPE"] = "en_US.UTF-8"


# In[2]:

def get_data_pairs(file, ratio=0.1):
    pairs = []
    entries = open(file, 'r', encoding='utf8').read().strip().split('\n\n')
    num = int(len(entries) * ratio)
    
    for entry in entries:
        sentence, target = [], []
        for line in entry.split('\n'):
            if line.strip() == '': continue
                
            token, pos, bio = line.split('\t')
            if token == '-LRB-': token = '('
            elif token == '-RRB-': token = ')'
                
            sentence.append(token)
            target.append(bio)
        pairs.append((sentence, target))

    return pairs[num:], pairs[:num]


# In[3]:

dse_train, dse_test = get_data_pairs('./dataset/dse.txt')
ese_train, ese_test = get_data_pairs('./dataset/ese.txt')
dse_train.sort(key=lambda x: len(x[1]), reverse=True)
ese_train.sort(key=lambda x: len(x[1]), reverse=True)
dse_test.sort(key=lambda x: len(x[1]), reverse=True)
ese_test.sort(key=lambda x: len(x[1]), reverse=True)

print(len(dse_train), len(ese_train))


# In[4]:

def sequence_to_ixs(seq, to_ix):
    ixs = [to_ix[w] if w in to_ix else to_ix[UNK_TOKEN] for w in seq]
    return torch.cuda.LongTensor(ixs)
#     return ixs


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


# for gensim word2vec
# def sequence_to_ixs2(seq):
#     vocabs = word_vectors.vocab.keys()
#     ixs = [word_vectors.vocab[w].index if w in vocabs else 0 for w in seq]
#     tensor = torch.cuda.LongTensor(ixs)
    
#     return autograd.Variable(tensor)


# In[5]:

# import gensim
# word_vectors = gensim.models.KeyedVectors.load_word2vec_format('/scepter/word_vectors/GoogleNews-vectors-negative300.bin', binary=True)  
# word_vectors.syn0


# In[6]:

class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, 
                 vocab_size, tagset_size, 
                 dropout, num_layers, bidirectional):
        super(LSTMTagger, self).__init__()
        
        self.direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim // self.direction
        self.num_layers = num_layers
        self.tagset_size = tagset_size

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        weights = torch.cuda.FloatTensor(embedding_weights)
        self.word_embeddings = nn.Embedding.from_pretrained(weights, freeze=True)
        
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, 
                            dropout=dropout, num_layers=self.num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
 
        self.hidden2tag = nn.Linear(self.hidden_dim * self.direction, self.tagset_size)
    
        # self.hidden = self.init_hidden()
 

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(self.num_layers * self.direction, batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.randn(self.num_layers * self.direction, batch_size, self.hidden_dim).cuda()))
 

    def forward(self, sentence, lengths):
        batch_size, seq_len = sentence.shape
        self.hidden = self.init_hidden(batch_size)
        
        try:
            embeds = self.word_embeddings(sentence) # [batch_size, seq_len, emb_dim]
            # embeds = embeds.view(seq_len, batch_size, -1)
            embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
            
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)
            lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True)

            tag_space = self.hidden2tag(lstm_out.contiguous().view(batch_size * seq_len, -1))
            tag_scores = F.log_softmax(tag_space, dim=1)
        
#             lstm_out = lstm_out.contiguous()
#             tag_space = self.hidden2tag(lstm_out.view(batch_size, -1, lstm_out.shape[2]))
#             tag_scores = F.log_softmax(tag_space, dim=2)
#             tag_scores = tag_scores.view(batch_size, -1, self.tagset_size)

            return tag_scores
        
        except Exception as e:
            print(sentence.shape)
            print(embeds.shape)
            print(e)


# In[7]:

def get_vectors(file, maximum=50000000000):
    glove, embedding_weights = {}, []
    ix, word_to_ix, ix_to_word = 0, {}, {}

    for line in open(file, 'r', encoding='utf8').readlines():
        line = line.strip().split(' ')
        if len(line) != (embedding_dim + 1): continue
        if line[0] in glove: continue

        vec = np.array(line[1:]).astype(np.float32)
        glove[line[0]] = vec
        embedding_weights.append(vec)
            
        word_to_ix[line[0]] = ix
        ix_to_word[ix] = line[0]
        ix += 1
        
        if ix > maximum: break

    glove[UNK_TOKEN] = [0] * len(embedding_weights[0])
    embedding_weights.append([0] * len(embedding_weights[0]))
    word_to_ix[UNK_TOKEN] = ix
    ix_to_word[ix] = UNK_TOKEN
    
    ix += 1
    glove[PAD_TOKEN] = [0] * len(embedding_weights[0])
    embedding_weights.append([0] * len(embedding_weights[0]))
    word_to_ix[PAD_TOKEN] = ix
    ix_to_word[ix] = PAD_TOKEN
    
    assert len(glove) == len(embedding_weights)
    
    print(len(glove))
    
    return glove, embedding_weights, word_to_ix, ix_to_word


# In[55]:

UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'

embedding_dim = 300
hidden_dim = 100
learning_rate = 0.01
momentum = 0.7
dropout = 0
num_layers = 3
bidirectional = True
batch_size = 1
epochs = 200
vector_file= 'dataset/glove/glove.840B.300d.txt'
model_path = 'models/standard.model'

train_data = dse_train
test_data = dse_test


# In[10]:

# glove, embedding_weights, word_to_ix, ix_to_word = get_vectors(vector_file)
# embedding_weights = np.array(embedding_weights).astype(np.float32)

# with open('dataset/glove.pickle', 'wb') as handle:
#     pickle.dump( [glove, embedding_weights, word_to_ix, ix_to_word] , handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dataset/glove.pickle', 'rb') as handle:
    glove, embedding_weights, word_to_ix, ix_to_word = pickle.load(handle)


tag_to_ix = {"B": 0, "I": 1, "O": 2, PAD_TOKEN: 3} # 手工設定詞性標籤數據字典
ix_to_tag = {0: "B", 1: "I", 2: "O", 3: PAD_TOKEN}


# In[56]:

model = LSTMTagger(embedding_dim, hidden_dim, 
                   len(word_to_ix), len(tag_to_ix), 
                   dropout=dropout,
                   num_layers=num_layers,
                   bidirectional=bidirectional)

loss_function = nn.NLLLoss()

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)

if torch.cuda.is_available():
    model.cuda()


# In[52]:

# TRAIN

total_num = len(train_data)
batch_num = math.ceil(total_num / batch_size)

for epoch in range(epochs):
    
    # for sentence, tags in train_data: 
    for i in range(batch_num):
        model.zero_grad()
        # model.hidden = model.init_hidden()
        
        data = train_data[i * batch_size : (i+1) * batch_size]
        
        # iterate: padding -> to_ix -> list
        # max_size = len(data[-1][0]) # get last one size in each batch
        # x = list(map(lambda x: sequence_to_ixs(padding(x[0], max_size), word_to_ix), data))
        # y = list(map(lambda x: sequence_to_ixs(padding(x[1], max_size), tag_to_ix), data))
        
        x = list(map(lambda pair: sequence_to_ixs(pair[0], word_to_ix), data))
        y = list(map(lambda pair: sequence_to_ixs(pair[1], tag_to_ix), data))
        
        assert len(x) == len(y)
        
        # x = autograd.Variable(torch.cuda.LongTensor(x))
        # y = autograd.Variable(torch.cuda.LongTensor(y))

        lengths = list(map(lambda x: x.shape[0], x))
        
        padded_seqs = pad_sequence(x, batch_first=True)
        padded_tags = pad_sequence(y, batch_first=True)
        
        predict_tags = model(padded_seqs, lengths)
        true_tags = padded_tags.view(-1)
        
        # one vs one
        # sentence_in = sequence_to_ixs(sentence, word_to_ix)
        # targets = sequence_to_ixs(tags, tag_to_ix)
        # tag_scores = model(sentence_in)

        loss = loss_function(predict_tags, true_tags)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 5 == 0:
        print("epoch: {}, loss: {}".format(epoch+1, loss))
        torch.save(model.state_dict(), model_path)


# In[57]:

model.load_state_dict(torch.load(model_path))


# In[58]:

def get_segments(tag_seq):
    segs = []
    start = -1
    for i, y in enumerate(tag_seq):
        if y == tag_to_ix["O"]: 
            if start != -1: segs.append((start, i))
            start = -1
        elif y == tag_to_ix["B"]:
            if start != -1: segs.append((start, i))
            start = i
        elif y == tag_to_ix["I"]:
            if start == -1: start = i
        else:
            print(y)
    
    if start != -1 and start != len(tag_seq):
        segs.append((start, len(tag_seq)))
        
    return segs


def show(y_predict, y_true):
    ps = [ix_to_tag[ix] for ix in y_predict.cpu().numpy()]
    ts = [ix_to_tag[ix] for ix in y_true.cpu().numpy()]
    
    print("Predict: {}\tTrue: {}".format(''.join(ps), ''.join(ts)))


def evaluate(predicts, trues):
    assert len(predicts) == len(trues)
    
    precision_prop, recall_prop = .0, .0
    precision_bin, recall_bin = 0, 0
    predict_total, true_total = 0, 0
    
    for y_predict, y_true in zip(predicts, trues):
        assert len(y_predict) == len(y_true)

        predict_segs = get_segments(y_predict)
        true_segs = get_segments(y_true)

        predict_count = len(predict_segs)
        true_count = len(true_segs)
        
        predict_total += predict_count
        true_total += true_count
        
        predict_flags = [False for i in range(predict_count)]
        true_flags = [False for i in range(true_count)]

        for t_i, (t_start, t_end) in enumerate(true_segs):
            for p_i, (p_start, p_end) in enumerate(predict_segs):
                assert p_start != p_end

                l_max = t_start if t_start > p_start else p_start
                r_min = t_end   if t_end   < p_end else p_end
                overlap = (r_min - l_max) if r_min > l_max else 0
                
                precision_prop += overlap / (p_end - p_start)
                recall_prop += overlap / (t_end - t_start)

                if not predict_flags[p_i] and overlap > 0:
                    precision_bin += 1
                    predict_flags[p_i] = True
                if not true_flags[t_i] and overlap > 0:
                    recall_bin += 1
                    true_flags[t_i] = True

                    
        # show(y_predict, y_true)
        
    precision = (precision_bin / predict_total) if predict_total != 0 else 1
    recall = recall_bin / true_total
    f1 = (2 * precision * recall) / (precision + recall)    
    binary_overlap = { 'precision': precision, 'recall': recall, 'f1': f1 }
    
    precision = (precision_prop / predict_total) if predict_total != 0 else 1
    recall = recall_prop / true_total
    f1 = (2 * precision * recall) / (precision + recall)
    proportional_overlap = { 'precision': precision, 'recall': recall, 'f1': f1 }
    
    print("Test data length: {}".format(len(predicts)))
    print("Precision\tBin: {}, Prop: {:.2f}, Predict Total: {}".format(precision_bin, precision_prop, predict_total))
    print("Recall\t\tBin: {}, Prop: {:.2f}, Recall Total: {}".format(recall_bin, recall_prop, true_total))
    print("=" * 75)
    print("Binary Overlap\t\tPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}".format(**binary_overlap))
    print("Proportional Overlap\tPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}".format(**proportional_overlap))
    
    return { 'binary': binary_overlap, 'proportional': proportional_overlap }


# In[59]:

# TEST
with torch.no_grad():
    data = test_data
    
    x = list(map(lambda pair: sequence_to_ixs(pair[0], word_to_ix), data))
    y = list(map(lambda pair: sequence_to_ixs(pair[1], tag_to_ix), data))

    lengths = list(map(lambda x: x.shape[0], x))

    padded_seqs = pad_sequence(x, batch_first=True)
    y_predicts = model(padded_seqs, lengths)
    y_predicts = torch.max(y_predicts, 1)[1].view([len(lengths), -1]) # .cpu().numpy()
    
    y_trues = y
    y_predicts = [y_[:lengths[i]] for i, y_ in enumerate(y_predicts)]

#     y_predicts, y_trues = [], []
#     for i in range(batch_num):
#         data = test_data[i * batch_size : (i+1) * batch_size]
            
#         seq, true_targets = each
#         inputs = autograd.Variable(torch.cuda.LongTensor(sequence_to_ixs(seq, word_to_ix)))
#         inputs = inputs.unsqueeze(0)
#         predict_targets = model(inputs)        
#         predict_targets = torch.max(predict_targets, 1)[1].cpu().numpy()
#         predict_targets = ixs_to_sequence(predict_targets, ix_to_tag)

#         y_predicts.append(predict_targets)
#         y_trues.append(true_targets)

    # 感覺可以實驗 tag by tag
    evaluate(y_predicts, y_trues)


# In[ ]:



