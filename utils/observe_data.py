import math

file_name = 'ese'

entries = open(f'dataset/{file_name}.txt', 'r', encoding='utf8').read().strip().split('\n\n')


def get_sent_and_tags(entry):
    sents, poss, tags = [], [], []
    
    lines = entry.strip().split('\n')
    for line in lines:            
        word, pos, tag = line.split('\t')
        length = len(word)
        sents.append(word)
        poss.append('{:>{length}s}'.format(pos, length=length))
        tags.append('{:>{length}s}'.format(tag, length=length))
        
    return sents, poss, tags


ws = open(f'dataset/obs_{file_name}.txt', 'w', encoding='utf8')

for entry in entries:
    sents, poss, tags = get_sent_and_tags(entry)
    
    print(' '.join(sents), file=ws)
    print(' '.join(tags), file=ws)
    print('='*20, file=ws)

ws.close()

