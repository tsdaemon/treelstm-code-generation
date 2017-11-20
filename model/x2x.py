import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import torch.nn.init as init

import Constants
from model.encoder import ChildSumTreeLSTM
from model.decoder import *
from model.layers import *


sys.setrecursionlimit(50000)


class Tree2TreeModel(nn.Module):
    def __init__(self, config, word_embeds):
        super(Tree2TreeModel).__init__(self)

        self.word_embedding = word_embeds

        self.encoder = ChildSumTreeLSTM(config.word_embed_dim, config.encoder_hidden_dim, config.dropout)

        self.decoder = CondAttLSTM(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                   config.decoder_hidden_dim,
                                   config.encoder_hidden_dim,
                                   config.attention_hidden_dim,
                                   config,
                                   config.dropout)

        self.src_ptr_net = PointerNet(config)

        self.terminal_gen_softmax = SoftmaxDense(config.decoder_hidden_dim, 2)
        init.uniform(self.terminal_gen_softmax.weight)

        self.rule_embedding_W = torch.FloatTensor((config.rule_num, config.rule_embed_dim))
        init.normal(self.rule_embedding_W, 0, 0.1)
        self.rule_embedding_b = torch.FloatTensor(config.rule_num)

        self.node_embedding = torch.FloatTensor((config.node_num, config.node_embed_dim))
        init.normal(self.node_embedding, 0, 0.1)

        self.vocab_embedding_W = torch.FloatTensor((config.target_vocab_size, config.rule_embed_dim))
        init.normal(self.vocab_embedding_W, 0, 0.1)
        self.vocab_embedding_b = torch.FloatTensor(config.target_vocab_size, name='vocab_embedding_b')

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = nn.Linear(config.decoder_hidden_dim, config.rule_embed_dim)
        init.uniform(self.decoder_hidden_state_W_rule.weight)
        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_token = nn.Linear(config.decoder_hidden_dim + config.encoder_hidden_dim,
                                                      config.rule_embed_dim)
        init.uniform(self.decoder_hidden_state_W_token.weight)

    def forward(self, tree, query):
        query_embed = self.word_embedding(query)
        h, ctx = self.encoder(tree, query_embed)




