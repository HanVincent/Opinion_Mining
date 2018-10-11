
# coding: utf-8

# In[1]:

import torch
import torch.autograd as autograd # torch中自動計算梯度模塊
import torch.nn as nn             # 神經網絡模塊
import torch.nn.functional as F   # 神經網絡模塊中的常用功能 
import torch.optim as optim       # 模型優化器模塊

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
            sentence.append(token)
            target.append(bio)
        pairs.append((sentence, target))

    return pairs[num:], pairs[:num]


# In[3]:

dse_train, dse_test = get_data_pairs('./dse.txt')
ese_train, ese_test = get_data_pairs('./ese.txt')
print(len(dse_train), len(ese_train))


# In[4]:

def get_dict(pair_data):
    # not normalized
    word_to_ix = {"_UNK": 0, "_PAD": 1} 

    for sent, tags in pair_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2} # 手工設定詞性標籤數據字典

    return word_to_ix, tag_to_ix

word_to_ix, tag_to_ix = get_dict(dse_train)
print(len(word_to_ix))


# In[5]:

class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, 
                 vocab_size, tagset_size, 
                 dropout, num_layers, bidirectional):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
 
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            dropout=dropout, num_layers=num_layers,
                            bidirectional=bidirectional)
 
        self.hidden2tag = nn.Linear(hidden_dim * (1+int(bidirectional)), tagset_size)
    
        self.hidden = self.init_hidden()
 

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers * (1+int(bidirectional)), 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers * (1+int(bidirectional)), 1, self.hidden_dim).cuda()))
 

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        
        try:
            lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)

            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

            tag_scores = F.log_softmax(tag_space)

            return tag_scores
        except Exception as e:
            print(sentence)
            print(embeds)
            print(e)


# In[6]:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    tensor = torch.cuda.LongTensor(idxs)
    
    return autograd.Variable(tensor)


# In[7]:

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
learning_rate = 0.1
dropout = 0
num_layers = 1
bidirectional = False
model_path = 'lstm_128.model'
epochs = 300

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 
                   len(word_to_ix), len(tag_to_ix), 
                   dropout=dropout,
                   num_layers=num_layers,
                   bidirectional=bidirectional)

loss_function = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model.cuda()


# In[ ]:

# TRAIN

for epoch in range(epochs): # 我們要訓練300次，可以根據任務量的大小酌情修改次數。
    for sentence, tags in dse_train:
        
        # 清除網絡先前的梯度值，梯度值是Pytorch的變量才有的數據，Pytorch張量沒有
        model.zero_grad()
        
        # 重新初始化隱藏層數據，避免受之前運行代碼的干擾
        model.hidden = model.init_hidden()
        
        # 準備網絡可以接受的的輸入數據和真實標籤數據，這是一個監督式學習
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        
        # 運行我們的模型，直接將模型名作為方法名看待即可
        tag_scores = model(sentence_in)
        
        # 計算損失，反向傳遞梯度及更新模型參數
        loss = loss_function(tag_scores, targets)
        
        loss.backward()
        
        optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss))
        torch.save(model.state_dict(), model_path)


# In[ ]:

# model.load_state_dict(torch.load(model_path))


# In[ ]:

# # TEST
# wrong = 0
# for each in dse_test:
#     seq, true_targets = each
#     inputs = prepare_sequence(seq, word_to_ix)
#     predict_targets = model(inputs)
#     predict_targets = torch.max(predict_targets, 1)[1].cpu().numpy()
    
#     for y, y_ in zip(true_targets, predict_targets):
#         if tag_to_ix[y] != y_:
#             print(seq)
#             print(true_targets)
#             print(predict_targets)
#             wrong += 1
#             break

# print("ratio:", wrong/len(dse_test))


# In[ ]:



