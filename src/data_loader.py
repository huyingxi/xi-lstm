from gensim.models import word2vec
from nltk import FreqDist
import numpy as np


def load_data_old(source, dist, max_len, vocab_size):
    '''
    doc me!
    '''
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [[i for i in x.split(' ')] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
        len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]
    X_max = max(map(len,X))

    y = [[j for j in y.split(' ')] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
        len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]

    for index in range(len(X)):
        round = X_max - len(X[index])
        while(round):
            X[index].append('.')
            y[index].append('O')
            round -= 1

    model = word2vec.Word2Vec.load('data/word2vec/mode.bin')

    words = list(model.wv.vocab)
    X_ix_to_word = words
    X_ix_to_word.append('UNK')
    X_word_to_ix = {word: ix for ix, word in enumerate(X_ix_to_word)}

    weight = []
    for i in range(len(X_ix_to_word)):
        if X_ix_to_word[i] in model.wv.vocab:
            weight_item = model[X_ix_to_word[i]].tolist()
            weight.append(weight_item)
        else:
            weight.append(np.random.randn(300, ).tolist())
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size - 1)

    count_in = 0
    count_out = 0
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):

            if word in X_word_to_ix:
                count_in += 1
                X[i][j] = X_word_to_ix[word]
            else:
                count_out += 1
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]
    y_ix_to_word.append('UNK')
    y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
    count_in = 0
    count_out = 0
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                count_in += 1
                y[i][j] = y_word_to_ix[word]
            else:
                count_out += 1
                y[i][j] = y_word_to_ix['UNK']

    return (
        X,
        len(X_word_to_ix), X_word_to_ix, X_ix_to_word,
        y,
        len(y_word_to_ix), y_word_to_ix, y_ix_to_word,
        weight,
    )


def load_data(source, dist, word_index, embedding_weight, max_len):
    '''
    doc me!
    '''
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [[i for i in (x.split(' '))] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
         len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]
    X_max = max(map(len,X))
    y = [[j for j in (y.split(' '))] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
        len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]

    # UNKNOWN
    word_index['UNK'] = len(word_index)

    b = np.random.rand(1, 300)
    # print(type(embedding_weight))
    # print(len(b))
    np.append(embedding_weight, b, axis=0)
    index_word = {word: ix for ix, word in enumerate(word_index)}

    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in word_index:
                X[i][j] = word_index[word]
            else:
                X[i][j] = word_index['UNK']

    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common()
    # { C: 3, D: 4 }
    y_ix_to_word = [word[0] for word in y_vocab]
    y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]

    seq_lengths = list(map(len, X))

    for i, sentence in enumerate(X):
        round = X_max - len(X[i])
        while(round):
            # X[i].append(0)
            y[i].append(0)
            round -= 1

    # import ipdb; ipdb.set_trace()
    return (
        X, word_index, index_word,
        y, y_word_to_ix, y_ix_to_word,
        embedding_weight,
        seq_lengths,
    )
