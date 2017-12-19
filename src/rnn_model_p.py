'''rnn model peephole
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
from xibase import (
    LSTMP,
)


class RNNModelP(nn.Module):
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
            return_sequences=True,
            bias=True,
            grad_clip=None,
            bidirectional=True
    ):
        super(RNNModelP, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_size = recurrent_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.rnn = LSTMP(input_size, hidden_size, recurrent_size, num_layers=num_layers, bias=bias, return_sequences=return_sequences, grad_clip=grad_clip, bidirectional=bidirectional)
        # self.fc = nn.Linear(recurrent_size, num_classes, bias=bias)

    def forward(self, x, lengths):
        # Set initial states
        zeros_h = Variable(torch.zeros(64, self.recurrent_size))
        zeros_c = Variable(torch.zeros(64, self.hidden_size))
        initial_states = [[(zeros_h, zeros_c)] * self.num_layers] * self.num_directions

        # Forward propagate RNN
        out = self.rnn(x, initial_states, lengths)
        # out, _ = self.rnn(x, initial_states=None)

        # Decode hidden state of last time step
        # out = self.fc(out)
        return out
