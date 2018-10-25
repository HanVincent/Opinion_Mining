def get_vectors(source, maximum=50000000000):
    word_vectors, embedding_weights = {}, []
    ix, word_to_ix, ix_to_word = 0, {}, {}

    if source == 'word2vec':
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin', binary=True)  
        
        for vocab in model.vocab:
            if vocab in word_vectors: continue

            vec = model.word_vec(vocab)
            word_vectors[vocab] = vec
            embedding_weights.append(vec)

            word_to_ix[vocab] = ix
            ix_to_word[ix] = vocab
            ix += 1

            if ix > maximum: break
    
    elif source == 'glove':
        for line in open('dataset/glove/glove.840B.300d.txt', 'r', encoding='utf8').readlines():
            line = line.strip().split(' ')
            if len(line) != (embedding_dim + 1): continue
            if line[0] in word_vectors: continue

            vec = np.array(line[1:]).astype(np.float32)
            word_vectors[line[0]] = vec
            embedding_weights.append(vec)

            word_to_ix[line[0]] = ix
            ix_to_word[ix] = line[0]
            ix += 1

            if ix > maximum: break

    word_vectors[UNK_TOKEN] = np.zeros(len(embedding_weights[0]))
    embedding_weights.append(np.zeros(len(embedding_weights[0])))
    word_to_ix[UNK_TOKEN] = ix
    ix_to_word[ix] = UNK_TOKEN
    
    assert len(word_vectors) == len(embedding_weights)
    
    print("Word vector size: {}".format(len(word_vectors)))
    
    return word_vectors, embedding_weights, word_to_ix, ix_to_word


UNK_TOKEN = '<UNK>'
source = 'word2vec'

word_vectors, embedding_weights, word_to_ix, ix_to_word = get_vectors(source)

with open(f'dataset/{source}.pickle', 'wb') as handle:
    pickle.dump( [word_vectors, embedding_weights, word_to_ix, ix_to_word] , handle, protocol=pickle.HIGHEST_PROTOCOL)