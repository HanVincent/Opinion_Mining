
# coding: utf-8

# In[ ]:

from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.optim as optim       # 模型優化器模塊

import numpy as np
import pickle, math, datetime, time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

is_cuda = torch.cuda.is_available()


# In[ ]:

from tensorboardX import SummaryWriter
writer = SummaryWriter('log')


# In[ ]:

from utils.preprocess import get_sentence_target, group_data, split_dataset


# In[ ]:

# Stacked BiLSTM
# from LSTM.standard_LSTM import LSTMTagger


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
            
            writer.add_scalar('Train/Loss'.format(epoch), loss.data[0], epoch)


# In[ ]:

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

        result = evaluate(y_predicts, y_trues)
        
        return result, (y_predicts, y_trues)


# In[ ]:

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


# In[ ]:

### Get Word Embeddings
with open(f'dataset/{source}.pickle', 'rb') as handle:
    word_vectors, embedding_weights, word_to_ix, ix_to_word = pickle.load(handle)

### Manual Tag
tag_to_ix = {"B": 0, "I": 1, "O": 2}
ix_to_tag = {0: "B", 1: "I", 2: "O"}


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
                       hidden_dim, 
                       len(tag_to_ix), 
                       dropout=dropout,
                       num_layers=num_layers,
                       bidirectional=bidirectional)

    if is_cuda: model.cuda()
        
    loss_function = nn.NLLLoss()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)

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




# In[ ]:




# In[ ]:

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


# In[ ]:

# for name, param in model.named_parameters():
#     print( name, param.shape)
    
# total_param = sum(p.numel() for p in model.parameters())
# print(total_param)


# In[ ]:



