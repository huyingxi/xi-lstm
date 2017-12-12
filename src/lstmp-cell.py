'''
lstmp for xi
'''
import torch
import torch.nn as nn
import torch.nn._functions.rnn as rnn
from torch.autograd import Variable
import argparse
from nltk import FreqDist
import sys
import string
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import word2vec
import os
from torch.nn import Parameter
from torch.autograd import Function

from torch.nn.modules.rnn import LSTMP, LSTMO

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

if sys.version_info < (3,):
  maketrans = string.maketrans
else:
  maketrans = str.maketrans


def text_to_word_sequence(text,
                          filters=' \t\n',
                          lower=False, split=" "):
  '''
  doc me!
  '''
  if lower:
    text = text.lower()
  text = text.translate(maketrans(filters, split * len(filters)))
  seq = text.split(split)
  return [i for i in seq if i]

def load_data(source, dist, max_len, vocab_size):
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

  for index in range(len(X)):
    round = X_max - len(X[index])
    while(round):
      X[index].append('.')
      y[index].append('O')
      round -= 1

  model = word2vec.Word2Vec.load('/Users/test/Desktop/RE/mode.bin')

  words = list(model.wv.vocab)
  X_ix_to_word = words
  X_ix_to_word.append('UNK')
  X_word_to_ix = {word: ix for ix, word in enumerate(X_ix_to_word)}

  weight = []
  for i in range(len(X_ix_to_word)):
    if i in model.wv.vocab:
      weight.append(model[X_ix_to_word[i]])
    else:
      weight.append([np.random.randn(300, )])
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
      len(X_word_to_ix), X_word_to_ix, X_ix_to_word, y,
      len(y_word_to_ix), y_word_to_ix, y_ix_to_word, weight,
  )


def process_data(word_sentences, max_len, word_to_ix):
  '''
  doc me!
  '''
  # Vectorizing each element in each sequence
  sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
  for i, sentence in enumerate(word_sentences):
    for j, word in enumerate(sentence):
      sequences[i, j, word] = 1.
  return sequences

def prepare_sequence(seq, to_ix):
  '''
  doc me!
  '''
  idxs = map(lambda w: to_ix[w], seq)
  tensor = torch.LongTensor(idxs)
  tensor = idxs
  return autograd.Variable(tensor)

class RNNModel(nn.Module):
  '''
  doc me!
  '''
  def __init__(
      self,
      input_size,
      hidden_size,
      recurrent_size,
      num_layers,
      num_classes,
      return_sequences = True,
      bias=True,
      grad_clip=None,
      bidirectional=True
  ):
    super(RNNModel, self).__init__()
    self.hidden_size = hidden_size
    self.recurrent_size = recurrent_size
    self.num_layers = num_layers
    self.return_sequences = return_sequences
    self.bidirectional = bidirectional
    self.num_directions = 2 if bidirectional else 1

    self.rnn = LSTMP(input_size, hidden_size, recurrent_size, num_layers=num_layers, bias=bias, return_sequences=return_sequences, grad_clip=grad_clip, bidirectional=bidirectional)
    #self.fc = nn.Linear(recurrent_size, num_classes, bias=bias)

  def forward(self, x):
    # Set initial states
    zeros_h = Variable(torch.zeros(x.size(0), self.recurrent_size))
    zeros_c = Variable(torch.zeros(x.size(0), self.hidden_size))
    initial_states = [[(zeros_h, zeros_c)] * self.num_layers] * self.num_directions

    # Forward propagate RNN
    out = self.rnn(x, initial_states)
    #out, _ = self.rnn(x, initial_states=None)

    # Decode hidden state of last time step
    #out = self.fc(out)
    return out


class RNNModel_O(nn.Module):
  '''
  doc me!
  '''
  def __init__(
      self,
      input_size,
      hidden_size,
      recurrent_size,
      num_layers,
      num_classes,
      return_sequences = True,
      bias=True,
      grad_clip=None,
      bidirectional=False,
  ):
    super(RNNModel_O, self).__init__()
    self.hidden_size = hidden_size
    self.recurrent_size = recurrent_size
    self.num_layers = num_layers
    self.return_sequences = return_sequences
    self.bidirectional = bidirectional

    self.rnn = LSTMO(input_size, hidden_size, recurrent_size, num_layers=num_layers, bias=bias, return_sequences=return_sequences, grad_clip=grad_clip, bidirectional=bidirectional)
    self.num_directions = 2 if bidirectional else 1
    #self.fc = nn.Linear(recurrent_size, num_classes, bias=bias)

  def forward(self, x):
    '''
    doc me!
    '''
    # Set initial states
    zeros_h = Variable(torch.zeros(x.size(0), self.recurrent_size))
    zeros_c = Variable(torch.zeros(x.size(0), self.hidden_size))
    zeros_t = Variable(torch.zeros(x.size(0), self.hidden_size))
    initial_states = [[(zeros_h, zeros_c, zeros_t)] * self.num_layers] * self.num_directions

    # Forward propagate RNN
    out = self.rnn(x, initial_states)
    #out, _ = self.rnn(x, initial_states=None)

    # Decode hidden state of last time step
    #out = self.fc(out)
    return out


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
    self.word_embeddings.weight.data.copy_(torch.from_numpy(np.array(word_embed_weight)))
    self.dropout = torch.nn.Dropout(0.5)

    self.lstmp = RNNModel(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        recurrent_size=hidden_dim,
        num_layers=1,
        num_classes=106,
        return_sequences=True,
        bias=True,
        grad_clip=10,
        bidirectional=True,
    )
    self.lstmo = RNNModel_O(
        input_size=embedding_dim,
        hidden_size=2*hidden_dim,
        recurrent_size=2*hidden_dim,
        num_layers=1,
        num_classes=106,
        return_sequences=True,
        bias=True,
        grad_clip=10,
        bidirectional=False,
    )
    self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size, bias=True)
    self.softmax = nn.Softmax()


  def forward(self, sentence):
    '''
    doc me!
    '''
    embeds = self.word_embeddings(sentence)
    embeds = self.dropout(embeds)
    embeds = self.lstmp(embeds)[0]
    embeds = self.lstmo(embeds)[0]
    tag_space = self.hidden2tag(embeds)
    tag_scores = F.softmax(tag_space, dim=-1)

    return tag_scores


class LossFun(nn.Module):
  '''
  doc me!
  '''
  def __init__(self, beta):
    '''
    doc me!
    '''
    super(LossFun, self).__init__()
    self.beta = beta
    return

  def forward(self, targets_scores, targets_in):
    loss = Variable(torch.zeros(1))
    max_index = torch.max(targets_scores, 2)[1] #(64,188)
    a = targets_in.data
    a = a.numpy()
    size = len(a)

    for batch in range((targets_in).size()[0]): # batch loop
      for length in range((targets_in[0].size()[0])): # words loop
        if torch.equal(max_index[batch][length], targets_in[batch][length]):
          if torch.equal(targets_in[batch][length].data, torch.LongTensor(1).zero_()):
            loss -= torch.log(targets_scores[batch][length][max_index[batch][length]])
          else:
            loss -= self.beta * torch.log(targets_scores[batch][length][max_index[batch][length]])

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


def predict(X, y, model):
  model.zero_grad()
  sentence = np.asarray(X)
  tensor = torch.from_numpy(sentence)
  sentence_in = autograd.Variable(tensor)
  tags = np.asarray(y)
  targets = torch.from_numpy(tags)
  targets_in = autograd.Variable(targets)
  tag_scores = model(sentence_in)

  return tag_scores, targets_in


def run():
  '''
  doc me!
  '''
  X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, word_embed_weight = load_data(
      '/Users/test/Desktop/RE/data/originaldata_new/train_test/train_x_real_filter.txt',
      '/Users/test/Desktop/RE/data/originaldata_new/train_test/train_y_real_filter.txt',
      MAX_LEN,
      VOCAB_SIZE,
  )
  model = LSTMTagger(EMBED_DIM, HIDDEN_DIM, len(X_word_to_ix), len(y_word_to_ix), word_embed_weight)
  print(model)

  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name, param.data)

  #loss_function = nn.NLLLoss()
  loss_function = LossFun(beta=10)
  #accuracy_function = AccuracyFun()
  optimizer = optim.RMSprop(
      model.parameters(),
      lr=0.1,
      alpha=0.99,
      eps=1e-08,
      weight_decay=0,
      momentum=0,
      centered=False,
  )

  f = open('/Users/test/Desktop/RE/data/originaldata_new/train_test/train_x_real_filter.txt', 'r')
  f1 = open('/Users/test/Desktop/RE/data/originaldata_new/train_test/train_y_real_filter.txt', 'r')
  X_test_data = f.read()
  Y_test_data = f1.read()
  f.close()
  f1.close()
  test_x = [text_to_word_sequence(x_)[::-1] for x_ in X_test_data.split('\n') if
            len(x_.split(' ')) > 0 and len(x_.split(' ')) <= MAX_LEN]
  test_y = [text_to_word_sequence(y_)[::-1] for y_ in Y_test_data.split('\n') if
            len(y_.split(' ')) > 0 and len(y_.split(' ')) <= MAX_LEN]

  X_max_test = max(map(len, test_x))
  for index in range(len(test_x)):
    round = X_max_test - len(test_x[index])
    while round:
      test_x[index].append('.')
      test_y[index].append('O')
      round -= 1

  for i, sentence in enumerate(test_x):
    for j, word in enumerate(sentence):
      if word in X_word_to_ix:
        test_x[i][j] = X_word_to_ix[word]
      else:
        test_x[i][j] = X_word_to_ix['UNK']

  for i, sentence in enumerate(test_y):
    for j, word in enumerate(sentence):
      if word in y_word_to_ix:
        test_y[i][j] = y_word_to_ix[word]
      else:
        test_y[i][j] = y_word_to_ix['UNK']

  count = 0

  log = open('/Users/test/Desktop/RE/log.txt', 'w')
  for epoch in xrange(NB_EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch : ", epoch)
    for i in range(len(X) - BATCH_SIZE):
      print("batch : ", i)
      optimizer.zero_grad()
      tag_scores, targets_in = predict(X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE], model)
      loss = loss_function(tag_scores, targets_in)
      loss.backward()
      print("current loss : ", loss.data)
      #acc = accuracy_function(tag_scores, targets_in)
      #print('accuracy : ', acc)
      # p1 = list(model.parameters())[0].clone()
      # optimizer.step()
      # p2 = list(model.parameters())[0].clone()
      # print(torch.equal(p1,p2))
      if count % 100 == 0:
        #torch.save(model, '/Users/test/Desktop/RE/model')
        print("{0} epoch , current training loss {1} : ".format(epoch, loss.data))
        log.write(str(epoch) + "epoch" + "current trainning loss : " + str(loss.data))
        test_scores, test_targets = predict(test_x[0:BATCH_SIZE], test_y[0:BATCH_SIZE], model)
        loss_test = loss_function(test_scores, test_targets)
        print(".............current test loss............ {} : ".format(loss_test/BATCH_SIZE))
        log.write("current test loss : " + str(loss_test/BATCH_SIZE))
      count += 1
    log.close()

run()
