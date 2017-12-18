'''
lstmp for xi
'''
import argparse
# import ipdb
import inspect
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
from torch import (
    LongTensor,
)

from typing import (
    Any,
    List,
    Tuple,
)

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
        a = targets_ground_truth
        a = a.numpy()
        size = len(a)

        for sentence_idx in range((targets_ground_truth).size()[0]):             # batch loop
            for word_idx in range((targets_ground_truth[0].size()[0])):     # words loop
                ground_truth_idx = targets_ground_truth[sentence_idx][word_idx]
                if length_matrix[sentence_idx][word_idx] == 1:
                    if ground_truth_idx == targets_ground_truth[sentence_idx][word_idx]:
                        if targets_ground_truth[sentence_idx][word_idx] == 0:  # take ['O'] 's value instead of zero
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


def accuracy_func(
        predict: Variable,
        ground_truth: LongTensor,
) -> float:
    total_num = len(torch.nonzero(ground_truth))

    ground_truth_modified = ground_truth.clone()
    ground_truth_modified[ground_truth == 0] = 1

    hit_tags = (torch.max(predict, 2)[1].view(ground_truth.size()).data == ground_truth_modified).sum()

    # a = targets_in.data
    # a = a.numpy()
    # size = len(a)
    return hit_tags/total_num


# class AccuracyFun(nn.Module):
#     '''
#     doc me!
#     '''
#     def __init__(self):
#         super(AccuracyFun, self).__init__()
#         return

#     def forward(self, targets_scores, targets_ground_truth):
#         total_tag_number = torch.nonzero(targets_ground_truth).size(0)
#         targets_in_clone = targets_ground_truth.clone()
#         targets_in_clone[targets_ground_truth==0] = 1     # two issue: 1 vs -1 & zero is for other???
#         hit_tags = (torch.max(targets_scores, 2)[1].view(targets_ground_truth.size()) == targets_in_clone).sum()

#         # a = targets_in.data
#         # a = a.numpy()
#         # size = len(a)
#         return hit_tags/total_tag_number


def predict(
        X: List,
        y: List,
        model: LSTMTagger,
        lengths: List[int],
) -> Tuple[Any, Any]:
    ''' predict
    '''
    # model.zero_grad()
    # no need to use Variable here. DELETE it.
    #
    # sentence = Variable(torch.zeros((len(X), 188)), requires_grad=False).long()
    sentence = torch.zeros((len(X), 188)).long()
    # sentence1 = torch.zeros((len(X), 188)).long()

    for idx, (seq, seqlen) in enumerate(zip(X, lengths)):
        sentence[idx, :seqlen] = torch.LongTensor(seq)

    lengths = torch.LongTensor(lengths)
    lengths, perm_idx = lengths.sort(0, descending=True)
    sentence = sentence[perm_idx]

    sentence = Variable(sentence, requires_grad=False)
    tag_scores = model(sentence, lengths)

    tags = np.asarray(y)
    targets = torch.from_numpy(tags)

    # targets_ground_truth = Variable(targets, requires_grad=False)     # delete it if possible
    targets_ground_truth = targets     # delete it if possible

    return tag_scores, targets_ground_truth


def main(args: dict) -> int:
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

    # import ipdb; ipdb.set_trace()
    c = list(zip(X, y, input_length))
    np.random.shuffle(c)
    X[:], y[:], input_length[:] = zip(*c)

    model = LSTMTagger(
        EMBED_DIM,
        HIDDEN_DIM,
        len(X_word_to_ix),
        len(y_word_to_ix),
        embedding_matrix_new,
    )

    # print(model)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    loss_function = LossFunc(beta=10)
    # accuracy_function = AccuracyFun()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

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

            print(type(tag_scores))
            print(type(targets_ground_truth))

            acc = accuracy_func(tag_scores, targets_ground_truth)
            print('accuracy : ', acc)
    log.close()

    return 0


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
    sys.exit(
        main(
            parse_arguments(
                sys.argv[1:]
            )
        )
    )
