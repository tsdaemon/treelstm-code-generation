import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import Constants
from model.encoder import ChildSumTreeLSTM


sys.setrecursionlimit(50000)


class Tree2TreeModel(nn.Module):
    def __init__(self, config):
        self.emb = nn.Embedding(config.source_vocab_size, config.word_embed_dim, padding_idx=Constants.PAD)
        self.encoder = ChildSumTreeLSTM(config.word_embed_dim, config.encoder_hidden_dim)

