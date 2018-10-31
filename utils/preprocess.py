from collections import defaultdict

LEFT_PARAENTHESIS = ['-LRB-', '-LSB-', '-LCB-']
RIGHT_PARAENTHESIS = ['-RRB-', '-RSB-', '-RCB-']

# dataset
SENTENCE_IDS = 'dataset/sentenceid.txt'
DOCS = 'dataset/datasplit/doclist.mpqaOriginalSubset'
TRAIN_DOCS = 'dataset/datasplit/filelist_train'
TEST_DOCS = 'dataset/datasplit/filelist_test'


def get_sentence_target(entry):
    sentence, target = [], []
    for line in entry.split('\n'):
        if line.strip() == '': continue
                
        token, pos, bio = line.split('\t')
    
        if token in LEFT_PARAENTHESIS: token = '('
        elif token in RIGHT_PARAENTHESIS: token = ')'
            
        sentence.append(token)
        target.append(bio)
        
    return sentence, target


def group_data(file):
    sentenceIDs = open(SENTENCE_IDS, 'r', encoding='utf8').read().strip().split('\n')
    entries = open(file, 'r', encoding='utf8').read().strip().split('\n\n')

    assert len(sentenceIDs) == len(entries)
    
    documents = defaultdict(lambda: [])
    for sent_id, entry in zip(sentenceIDs, entries):
        sent_id = sent_id.split(' ')[2]
        sentence, target = get_sentence_target(entry)        
        documents[sent_id].append((sentence, target))

    return documents


def split_dataset(documents, num):
    docs_list = open(DOCS, 'r', encoding='utf8').read().strip().split('\n')
    train_ids = open(TRAIN_DOCS + str(num), 'r', encoding='utf8').read().strip().split('\n')
    test_ids = open(TEST_DOCS + str(num), 'r', encoding='utf8').read().strip().split('\n')
    
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