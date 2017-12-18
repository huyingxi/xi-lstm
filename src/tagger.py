'''
module doc
'''
from typing import (
    Any,
    List,
    # Tuple,
)

import numpy as np
import torch
from torch import (
    nn,
    # LongTensor,
)
from torch.autograd import (
    Variable,
)
import torch.nn.functional as F

from rnn_model_p import (
    RNNModelP,
)
from rnn_model_o import (
    RNNModelO,
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

        self.lstmp = RNNModelP(
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
        self.lstmo = RNNModelO(
            input_size=2*hidden_dim,
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

    def forward(
            self,
            sentence: Variable,
            lengths: List[int],
    ) -> Any:
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout(embeds)
        embeds = self.lstmp(embeds, lengths)[0]
        embeds = self.lstmo(embeds, lengths)[2]
        tag_space = self.hidden2tag(embeds)
        tag_scores = F.softmax(tag_space, dim=-1)

        return tag_scores
