'''
lstmp for xi
'''
import argparse
# import ipdb
import os
import pickle
import sys
import string

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

from data_loader import (
    load_data,
)

from tagger import (
    LSTMTagger,
)


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


class LossFunc(nn.Module):
    '''
    doc me!
    '''
    def __init__(self, beta):
        '''
        doc me!
        '''
        super(LossFunc, self).__init__()
        self.beta = beta
        return

    def forward(self, targets_scores, targets_ground_truth, y_ix_to_word, lengths):
        length_matrix = np.zeros((64, 188))
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                length_matrix[i][j] = 1

        loss = Variable(torch.zeros(1))
        max_index = torch.max(targets_scores, 2)[1]     # (64,188)
        a = targets_ground_truth.data
        a = a.numpy()
        size = len(a)

        for sentence_idx in range((targets_ground_truth).size()[0]):             # batch loop
            for word_idx in range((targets_ground_truth[0].size()[0])):     # words loop
                ground_truth_idx = targets_ground_truth[sentence_idx][word_idx]
                if length_matrix[sentence_idx][word_idx] == 1:
                    if torch.equal(
                        ground_truth_idx,
                        targets_ground_truth[sentence_idx][word_idx]
                    ):
                        if torch.equal(
                            targets_ground_truth[sentence_idx][word_idx].data,
                            torch.LongTensor(1).zero_()  # take ['O'] 's value instead of zero
                        ):
                            loss -= torch.log(
                                targets_scores[sentence_idx][word_idx][ground_truth_idx]
                            )
                        else:
                            loss -= self.beta * torch.log(targets_scores[sentence_idx][word_idx][ground_truth_idx])
                    else:
                        loss -= torch.log(targets_scores[sentence_idx][word_idx][ground_truth_idx])
                        # ground_truth_idx = targets_ground_truth[sentence_idx][word_idx]
                        # diff = 1 - targets_scores[sentence_idx][word_idx][ground_truth_idx]

        return loss/size


class AccuracyFun(nn.Module):
    '''
    doc me!
    '''
    def __init__(self):
        super(AccuracyFun, self).__init__()
        return

    def forward(self, targets_scores, targets_in):
        total_tag_number = torch.nonzero(targets_in).size(0)
        targets_in_clone = targets_in.clone()
        targets_in_clone[targets_in==0] = 1     # two issue: 1 vs -1 & zero is for other???
        hit_tags = (torch.max(targets_scores, 2)[1].view(targets_in.size()).data == targets_in_clone.data).sum()

        # a = targets_in.data
        # a = a.numpy()
        # size = len(a)
        return hit_tags/total_tag_number


def predict(X, y, model, lengths):
    # model.zero_grad()
    # no need to use Variable here. DELETE it.
    #
    sentence = Variable(torch.zeros((len(X), 188)), requires_grad=False).long()
    for idx, (seq, seqlen) in enumerate(zip(X, lengths)):
        sentence[idx, :seqlen] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    lengths, perm_idx = lengths.sort(0, descending=True)
    sentence = sentence[perm_idx]

    tag_scores = model(sentence, lengths)

    tags = np.asarray(y)
    targets = torch.from_numpy(tags)
    targets_ground_truth = Variable(targets, requires_grad=False)     # delete it if possible

    return tag_scores, targets_ground_truth


def main(args):
    ''' Main entrypoint '''

    MAX_LEN = args.max_len
    VOCAB_SIZE = args.vocab_size
    BATCH_SIZE = args.batch_size
    LAYER_NUM = args.layer_num
    HIDDEN_DIM = args.hidden_dim
    NB_EPOCH = args.nb_epoch
    MODE = args.mode
    EMBED_DIM = args.embed_dim

#    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, word_embed_weight = load_data_old(
#        'data/train_test/train_x_real_filter.txt',
#        'data/train_test/train_y_real_filter.txt',
#        MAX_LEN,
#        VOCAB_SIZE,
#    )

    with open('data/word2vec_google300_for_NYT.pkl', 'rb') as vocab:
       word_index = pickle.load(vocab,encoding='latin1')
       embedding_matrix = pickle.load(vocab,encoding='latin1')

    X, X_word_to_ix, X_ix_to_word, y, y_word_to_ix, y_ix_to_word, embedding_weight, input_length = load_data(
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
    accuracy_function = AccuracyFun()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # optimizer = optim.RMSprop(
    #     model.parameters(),
    #     lr=0.1,
    #     alpha=0.99,
    #     eps=1e-08,
    #     weight_decay=0,
    #     momentum=0,
    #     centered=False,
    # )

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
        for i in range(0, (len(X) - 2*BATCH_SIZE), BATCH_SIZE):
            print("batch {0}, total_batch {1}: ".format(int(i/BATCH_SIZE), int(len(X)/BATCH_SIZE)))
            optimizer.zero_grad()
            tag_scores, targets_ground_truth = predict(
                X[i:i+BATCH_SIZE],
                y[i:i+BATCH_SIZE],
                model,
                input_length[i:i+BATCH_SIZE]
            )
            loss = loss_function(tag_scores, targets_ground_truth, y_ix_to_word, input_length[i:i+BATCH_SIZE])
            loss.backward()
            optimizer.step()

            print("current loss : ", loss.data)

            acc = accuracy_function(tag_scores, targets_ground_truth)
            print('accuracy : ', acc)
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


def parse_arguments(argv):
    """ args """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-max_len', type=int, default=200,
        help='max_len help message'
    )
    parser.add_argument('-vocab_size', type=int, default=45000)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-layer_num', type=int, default=1)
    parser.add_argument('-hidden_dim', type=int, default=300)
    parser.add_argument('-nb_epoch', type=int, default=5)
    parser.add_argument('-mode', default='train')
    parser.add_argument('-embed_dim', type=int, default=300)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
