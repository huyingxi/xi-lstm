'''
lstmp for xi
'''
import argparse
import sys
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import numpy as np

import torch
from torch import (
    LongTensor,
    optim,
)
from torch.autograd import (
    Variable,
)

from data_loader import (
    load_train_test_data,
)

from tagger import (
    LSTMTagger,
)


def loss_func(
        y_predict: Variable,
        y_truth: LongTensor,
        y_ix_to_word: List,
        lengths: List,
        *,
        beta: int=10,
) -> float:
    '''loss function
    '''
    length_matrix = np.zeros((64, 188))
    for i in range(len(lengths)):
        for j in range(lengths[i]):
            length_matrix[i][j] = 1

    loss = Variable(torch.zeros(1)).cuda()
    max_index = torch.max(y_predict, 2)[1]     # (64,188)
    # a = targets_ground_truth
    # a = a.numpy()
    size = len(y_truth)

    for sentence_idx in range((y_truth).size()[0]):             # batch loop
        for word_idx in range((y_truth[0].size()[0])):     # words loop
            ground_truth_idx = y_truth[sentence_idx][word_idx]
            predict_max_idx = max_index[sentence_idx][word_idx]
            if length_matrix[sentence_idx][word_idx] == 1:
                # import ipdb; ipdb.set_trace()
                if torch.equal(predict_max_idx.data, torch.LongTensor(np.array([ground_truth_idx])).cuda()):
                    # if predict_max_idx == torch.LongTensor(ground_truth_idx):
                    if ground_truth_idx == 0:  # take ['O'] 's value instead of zero
                        loss -= torch.log(
                            y_predict[sentence_idx][word_idx][predict_max_idx]
                        )
                    else:
                        loss -= beta * torch.log(y_predict[sentence_idx][word_idx][predict_max_idx])
                else:
                    loss -= torch.log(y_predict[sentence_idx][word_idx][ground_truth_idx])
                    # ground_truth_idx = targets_ground_truth[sentence_idx][word_idx]
                    # diff = 1 - targets_scores[sentence_idx][word_idx][ground_truth_idx]

    return loss/size


def accuracy_func(
        y_predict: Variable,
        y_truth: LongTensor,
) -> float:
    '''accuracy
    '''
    total_num = len(torch.nonzero(y_truth))

    y_truth_modified = y_truth.clone()
    y_truth_modified[y_truth == 0] = -1

    hit_tags = (torch.max(y_predict, 2)[1].view(y_truth.size()).data.cpu() == y_truth_modified).sum()

    # a = targets_in.data
    # a = a.numpy()
    # size = len(a)
    return hit_tags/total_num


def accuracy_func_test(
        y_predict: Variable,
        y_truth: LongTensor,
) -> float:
    '''
    accuracy test
    '''
    total_num = len(torch.nonzero(y_truth))

    ground_truth_modified = y_truth.clone()
    ground_truth_modified[y_truth == 0] = -1

    hit_tags = 0
    for i in range(len(y_truth)):  # batch loop
        for j in range(len(y_truth[i])):   # words loop
            if y_truth[i][j] != 0 :
                if(y_truth[i][j] in np.array(torch.sort(y_predict[i][j], -1, True)[1][:10].data)):  # if top 10 hits
                    hit_tags += 1

    return hit_tags/total_num


def predict_bak(
        X: Variable,
        model: LSTMTagger,
        lengths: List[int],
) -> Any:
    ''' predict
    '''
    # model.zero_grad()
    # no need to use Variable here. DELETE it.
    #
    # sentence = Variable(torch.zeros((len(X), 188)), requires_grad=False).long()
    # sentence = torch.zeros((len(X), 188)).long()
    # sentence1 = torch.zeros((len(X), 188)).long()

    # for idx, (seq, seqlen) in enumerate(zip(X, lengths)):
    #     sentence[idx, :seqlen] = torch.LongTensor(seq)

    # lengths = torch.LongTensor(lengths)
    # lengths, perm_idx = lengths.sort(0, descending=True)
    # sentence = sentence[perm_idx]

    # sentence = Variable(sentence, requires_grad=False)
    tag_scores = model(X, lengths)

    return tag_scores


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

    X, X_word_to_idx, X_ix_to_word, y, y_word_to_idx, y_ix_to_word, \
        embedding_weight, input_length = load_train_test_data()

    # import ipdb; ipdb.set_trace()
    c = list(zip(X, y, input_length))
    np.random.shuffle(c)
    X[:], y[:], input_length[:] = zip(*c)

    tagger = LSTMTagger(
        EMBED_DIM,
        HIDDEN_DIM,
        len(X_word_to_idx),
        len(y_word_to_idx),
        embedding_weight,
    ).cuda()

    sentence = torch.zeros((len(X), 188)).long()
    for idx, (seq, seqlen) in enumerate(zip(X, input_length)):
        sentence[idx, :seqlen] = torch.LongTensor(seq)
    sentence = Variable(sentence, requires_grad=False)
    X = sentence.cuda()

    # print(model)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    optimizer = optim.SGD(tagger.parameters(), lr=0.01)

    log = open('data/log.txt', 'w')

    # again, normally you would NOT do 300 epochs, it is toy data
    for epoch in range(NB_EPOCH):
        for i in range(0, (len(X) - 2*BATCH_SIZE), BATCH_SIZE):
            print("epoch[{}] batch[{}/{}] ".format(
                epoch,
                int(i/BATCH_SIZE),
                int(len(X)/BATCH_SIZE),
                ),
                  end='',
                 )
            optimizer.zero_grad()

            y_truth = torch.from_numpy(
                np.asarray(y[i:i+BATCH_SIZE])
            )

            y_predict = tagger(
                X[i:i+BATCH_SIZE],
                input_length[i:i+BATCH_SIZE]
            )
            # tag_scores = predict(
            #     X[i:i+BATCH_SIZE],
            #     model,
            #     input_length[i:i+BATCH_SIZE]
            # )

            loss = loss_func(
                y_predict,
                y_truth,
                y_ix_to_word,
                input_length[i:i+BATCH_SIZE],
            )

            loss.backward()
            optimizer.step()

            acc = accuracy_func(y_predict, y_truth)
            acc_test = accuracy_func_test(y_predict, y_truth)

            print('current loss[%4.2f] / '
                  'accuracy:[%4.2f] / '
                  'accuracy_test:[%4.2f]'
                  % (loss.data[0], acc, acc_test))

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
