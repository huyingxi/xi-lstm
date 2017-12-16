'''
Huan testing 123
'''
import argparse
# import ipdb
import os
import pickle
import sys
import string

from nltk import FreqDist
import numpy as np
import torch
import torch.nn as nn
import torch.nn._functions.rnn as rnn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.autograd import Function

from gensim.models import word2vec

# from xibase import (
#     LSTMO,
#     LSTMP,
# )

ap = argparse.ArgumentParser()
ap.add_argument('-max_len', type=int, default=200)
ap.add_argument('-vocab_size', type=int, default=45000)
ap.add_argument('-batch_size', type=int, default=64)
ap.add_argument('-layer_num', type=int, default=1)
ap.add_argument('-hidden_dim', type=int, default=300)
ap.add_argument('-nb_epoch', type=int, default=5)
ap.add_argument('-mode', default='train')
ap.add_argument('-embed_dim', type=int, default=300)
args = vars(ap.parse_args())

MAX_LEN = args['max_len']
VOCAB_SIZE = args['vocab_size']
BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']
EMBED_DIM = args['embed_dim']


def text_to_word_sequence(
        text,
        filters=' \t\n',
        lower=False,
        split=" ",
):
    '''
    doc me!
    '''
    if lower:
        text = text.lower()
    text = text.translate(str.maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]


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
    X_max = max(map(len, X))
    y = [[j for j in (y.split(' '))] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if
        len(x) > 0 and len(y) > 0 and len(x.split(' ')) <= max_len and len(y.split(' ')) <= max_len]

    word_index['UNK'] = len(word_index)

    b = np.random.rand(1,300)
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
    y_ix_to_word = [word[0] for word in y_vocab]
    y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]

    seq_lengths = []
    seq_lengths = (map(len, X))

    for i, sentence in enumerate(X):
        round = X_max - len(X[i])
        while round:
            # X[i].append(0)
            y[i].append(0)
            round -= 1

    return (
        X,
        word_index,
        index_word,
        y,
        y_word_to_ix,
        y_ix_to_word,
        embedding_weight,
        seq_lengths,
    )


class LSTMTagger(nn.Module):
    '''
    doc me!
    '''
    def __init__(
            self,
            embedding_dim,
            hidden_dim,
            vocab_size,
            tagset_size,
            word_embed_weight,
    ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # ipdb.set_trace()
        np_weight = np.array(word_embed_weight)
        weight = torch.from_numpy(np_weight)
        self.word_embeddings.weight.data.copy_(weight)

        self.dropout = torch.nn.Dropout(0.5)

        self.lstmp = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.lstmo = nn.LSTM(
            input_size=2*hidden_dim,
            hidden_size=2*hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, sentence, _):
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)
        embeds = self.lstmp(embeds)[0]
        embeds = self.lstmo(embeds)[0]
        tag_space = self.hidden2tag(embeds)
        tag_scores = F.softmax(tag_space, dim=-1)

        return tag_scores


class LossFunc(nn.Module):
    '''
    doc me!
    '''
    def __init__(self, beta):
        super(LossFunc, self).__init__()
        self.beta = beta
        return

    def forward(self, targets_scores, targets_in, y_ix_to_word, lengths):
        length_matrix = np.zeros((64, 188))
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                length_matrix[i][j] = 1

        loss = Variable(torch.zeros(1))
        max_index = torch.max(targets_scores, 2)[1]     # (64,188)
        a = targets_in.data
        a = a.numpy()
        size = len(a)

        for batch in range((targets_in).size()[0]):             # batch loop
            for length in range((targets_in[0].size()[0])):     # words loop
                if length_matrix[batch][length] == 1:
                    if torch.equal(
                            max_index[batch][length],
                            targets_in[batch][length]
                    ):
                        if torch.equal(
                                targets_in[batch][length].data,
                                torch.LongTensor(1).zero_()
                        ):
                            loss -= torch.log(
                                targets_scores[batch][length][max_index[batch][length]]
                            )
                        else:
                            loss -= self.beta * torch.log(targets_scores[batch][length][max_index[batch][length]])
                    else:
                        loss -= torch.log(targets_scores[batch][length][targets_in[batch][length]])

        return loss/size


class AccuracyFun(nn.Module):
    '''
    doc me!
    '''
    def __init__(self):
        super(AccuracyFun, self).__init__()
        return

    def forward(self, targets_scores, targets_in):
        acc = (torch.max(targets_scores, 2)[1].view(targets_in.size()).data == targets_in.data).sum()
        a = targets_in.data
        a = a.numpy()
        size = len(a)
        return acc/size


def predict(X, y, model, lengths):
    '''
    predict
    '''
    model.zero_grad()
    sentence_in = Variable(torch.zeros((len(X), 188))).long()
    for idx, (seq, seqlen) in enumerate(zip(X, lengths)):
        sentence_in[idx, :seqlen] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    lengths, perm_idx = lengths.sort(0, descending=True)
    sentence_in = sentence_in[perm_idx]

    tag_scores = model(sentence_in, lengths)

    tags = np.asarray(y)
    targets = torch.from_numpy(tags)
    targets_in = autograd.Variable(targets)

    return tag_scores, targets_in


def run():
    '''
    doc me!
    '''
    with open('data/word2vec_google300_for_NYT.pkl', 'rb') as vocab:
        word_index = pickle.load(vocab, encoding='latin1')
        embedding_matrix = pickle.load(vocab, encoding='latin1')

    X, X_word_to_ix, X_ix_to_word, \
        y, y_word_to_ix, y_ix_to_word, \
        embedding_weight, input_length = load_data(
            'data/train_test/train_x_real_filter.txt',
            'data/train_test/train_y_real_filter.txt',
            word_index,
            embedding_matrix,
            max_len=188,
        )

    embedding_matrix_new = []
    for i in embedding_matrix:
        embedding_matrix_new.append(i)

    c = list(zip(X, y, input_length))
    np.random.shuffle(c)
    X[:], y[:], input_length = zip(*c)

    model = LSTMTagger(
        EMBED_DIM,
        HIDDEN_DIM,
        len(X_word_to_ix),
        len(y_word_to_ix),
        embedding_matrix_new,
    )
    print(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # loss_function = nn.NLLLoss()
    loss_function = LossFunc(beta=10)
    # accuracy_function = AccuracyFun()
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=1e-2,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False,
    )

#    f = open('data/train_test/train_x_real_filter.txt', 'r')
#    f1 = open('data/train_test/train_y_real_filter.txt', 'r')
#    X_test_data = f.read()
#    Y_test_data = f1.read()
#    f.close()
#    f1.close()
#    test_x = [text_to_word_sequence(x_)[::-1] for x_ in X_test_data.split('\n') if
#             len(x_.split(' ')) > 0 and len(x_.split(' ')) <= MAX_LEN]
#    test_y = [text_to_word_sequence(y_)[::-1] for y_ in Y_test_data.split('\n') if
#             len(y_.split(' ')) > 0 and len(y_.split(' ')) <= MAX_LEN]
#
#    X_max_test = max(map(len, test_x))
#    for index in range(len(test_x)):
#        round = X_max_test - len(test_x[index])
#        while round:
#            test_x[index].append('.')
#            test_y[index].append('O')
#            round -= 1
#
#    for i, sentence in enumerate(test_x):
#        for j, word in enumerate(sentence):
#            if word in X_word_to_ix:
#                test_x[i][j] = X_word_to_ix[word]
#            else:
#                test_x[i][j] = X_word_to_ix['UNK']
#
#    for i, sentence in enumerate(test_y):
#        for j, word in enumerate(sentence):
#            if word in y_word_to_ix:
#                test_y[i][j] = y_word_to_ix[word]
#            else:
#                test_y[i][j] = y_word_to_ix['UNK']

    count = 0

    log = open('data/log.txt', 'w')

    # again, normally you would NOT do 300 epochs, it is toy data
    for epoch in range(NB_EPOCH):
        print("epoch : ", epoch)
        for i in range(len(X) - BATCH_SIZE):
            print("batch : ", i)
            optimizer.zero_grad()
            tag_scores, targets_in = predict(
                X[i:i+BATCH_SIZE],
                y[i:i+BATCH_SIZE],
                model,
                input_length[i:i+BATCH_SIZE],
            )
            loss = loss_function(
                tag_scores,
                targets_in,
                y_ix_to_word,
                input_length[i:i+BATCH_SIZE],
            )
            loss.backward()
            print("current loss : ", loss.data)
            # acc = accuracy_function(tag_scores, targets_in)
            # print('accuracy : ', acc)
            # p1 = list(model.parameters())[0].clone()
            # optimizer.step()
            # p2 = list(model.parameters())[0].clone()
            # print(torch.equal(p1,p2))
#            if count % 100 == 0:
#                # torch.save(model, '/Users/test/Desktop/RE/model')
#                print("{0} epoch , current training loss {1} : ".format(epoch, loss.data))
#                log.write(str(epoch) + "epoch" + "current trainning loss : " + str(loss.data))
#                test_scores, test_targets = predict(
#                    test_x[0:BATCH_SIZE],
#                    test_y[0:BATCH_SIZE],
#                    model,
#                )
#                loss_test = loss_function(test_scores, test_targets)
#                print(".............current test loss............ {} : ".format(loss_test/BATCH_SIZE))
#                log.write("current test loss : " + str(loss_test/BATCH_SIZE))
#            count += 1
    log.close()


run()
