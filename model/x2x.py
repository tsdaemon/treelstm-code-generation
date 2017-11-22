import sys
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable as Var
import torch.nn.init as init

import Constants
from model.encoder import ChildSumTreeLSTM
from model.decoder import *
from model.layers import *
from model.utils import *


sys.setrecursionlimit(50000)


class Tree2TreeModel(nn.Module):
    def __init__(self, config, word_embeds):
        super().__init__()

        self.config = config
        self.cuda = config.cuda

        self.word_embedding = word_embeds
        if self.cuda:
            self.word_embedding = self.word_embedding.cuda()

        self.encoder = ChildSumTreeLSTM(config.word_embed_dim, config.encoder_hidden_dim, config.dropout)

        self.decoder = CondAttLSTM(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                   config.decoder_hidden_dim,
                                   config.encoder_hidden_dim,
                                   config.attention_hidden_dim,
                                   config,
                                   config.dropout)

        self.src_ptr_net = PointerNet(config)

        self.terminal_gen_softmax = SoftmaxDense(config.decoder_hidden_dim, 2)
        init.xavier_uniform(self.terminal_gen_softmax.weight)

        self.rule_embedding_W = Parameter(torch.FloatTensor(config.rule_num, config.rule_embed_dim))
        init.normal(self.rule_embedding_W, 0, 0.1)
        self.rule_embedding_b = Parameter(torch.FloatTensor(config.rule_num).zero_())
        # if self.cuda:
        #     self.rule_embedding_W = self.rule_embedding_W.cuda()
        #     self.rule_embedding_b = self.rule_embedding_b.cuda()

        self.node_embedding = Parameter(torch.FloatTensor(config.node_num, config.node_embed_dim))
        init.normal(self.node_embedding, 0, 0.1)
        # if self.cuda:
        #     self.node_embedding = self.node_embedding.cuda()

        self.vocab_embedding_W = Parameter(torch.FloatTensor(config.target_vocab_size, config.rule_embed_dim))
        init.normal(self.vocab_embedding_W, 0, 0.1)
        self.vocab_embedding_b = Parameter(torch.FloatTensor(config.target_vocab_size).zero_())
        # if self.cuda:
        #     self.vocab_embedding_W = self.vocab_embedding_W.cuda()
        #     self.vocab_embedding_b = self.vocab_embedding_b.cuda()

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = nn.Linear(config.decoder_hidden_dim, config.rule_embed_dim)
        init.xavier_uniform(self.decoder_hidden_state_W_rule.weight)
        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_token = nn.Linear(config.decoder_hidden_dim + config.encoder_hidden_dim,
                                                      config.rule_embed_dim)
        init.xavier_uniform(self.decoder_hidden_state_W_token.weight)

        self.log_softmax = nn.LogSoftmax()

    def forward(self, tree, query):
        pass

    def forward_train(self, tree, query, tgt_node_seq, tgt_action_seq, tgt_par_rule_seq, tgt_par_t_seq, tgt_action_seq_type):
        # prepare output
        # (max_example_action_num, node_embed_dim)
        tgt_node_embed = self.node_embedding[tgt_node_seq]

        # (max_example_action_num, rule_embed_dim)
        tgt_action_seq_embed = ifcond(tgt_action_seq[:, 0] > 0,
                                     self.rule_embedding_W[tgt_action_seq[:, 0]],
                                     self.vocab_embedding_W[tgt_action_seq[:, 1]])

        # parent rule application embeddings
        # (max_example_action_num, rule_embed_dim)
        tgt_par_rule_embed = Var(zeros(tgt_par_rule_seq.shape[0], self.config.rule_embed_dim, cuda=self.cuda))
        tgt_par_rule_embed[1:] = self.rule_embedding_W[tgt_par_rule_seq[1:]]

        # (max_example_action_num, rule_embed_dim)
        tgt_action_seq_embed_tm1 = Var(zeros_like(tgt_action_seq_embed, self.cuda))
        tgt_action_seq_embed_tm1[1:, :] = tgt_action_seq_embed[:-1, :]

        # get encoder results
        # (max_query_length, word_embed_dim)
        query_embed = Var(self.word_embedding[query], requires_grad=False)
        # (decoder_hidden_dim), (max_query_length, decoder_hidden_dim)
        h, ctx = self.encoder(tree, query_embed)

        # (max_example_action_num, rule_embed_dim + node_embed_dim + rule_embed_dim)
        decoder_input = torch.cat([tgt_action_seq_embed_tm1, tgt_node_embed, tgt_par_rule_embed], dim=-1)

        # (max_example_action_num, decoder_hidden_dim), (max_example_action_num, encoder_hidden_dim)
        decoder_hidden_states, ctx_vectors = self.decoder.forward_train(decoder_input,
                                                                        ctx, h, tgt_par_t_seq)

        # (max_example_action_num, decoder_hidden_state + encoder_hidden_dim)
        decoder_concat = torch.cat([decoder_hidden_states, ctx_vectors], dim=-1)

        # (max_example_action_num, rule_embed_dim)
        decoder_hidden_state_trans_rule = self.decoder_hidden_state_W_rule(decoder_hidden_states)
        decoder_hidden_state_trans_token = self.decoder_hidden_state_W_token(decoder_concat)

        # (max_example_action_num, rule_num)
        rule_predict = self.log_softmax(torch.addmm(self.rule_embedding_b, decoder_hidden_state_trans_rule, torch.t(self.rule_embedding_W)))

        # (max_example_action_num, 2)
        terminal_gen_action_prob = self.terminal_gen_softmax(decoder_hidden_states)

        # (max_example_action_num, target_vocab_size)
        vocab_predict = self.log_softmax(torch.addmm(self.vocab_embedding_b, decoder_hidden_state_trans_token, torch.t(self.vocab_embedding_W)))

        # (max_example_action_num, max_query_length)
        copy_prob = self.src_ptr_net(ctx, decoder_concat)

        # (max_example_action_num)
        rule_tgt_prob = rule_predict.gather(1, Var(tgt_action_seq[:, 0].unsqueeze(1), requires_grad=False))

        # (max_example_action_num)
        vocab_tgt_prob = vocab_predict.gather(1, Var(tgt_action_seq[:, 1].unsqueeze(1), requires_grad=False))

        # (max_example_action_num)
        copy_tgt_prob = copy_prob.gather(1, Var(tgt_action_seq[:, 2].unsqueeze(1), requires_grad=False))

        # (max_example_action_num)
        tgt_prob = Var(tgt_action_seq_type[:, 0].float(), requires_grad=False) * rule_tgt_prob + \
                   Var(tgt_action_seq_type[:, 1].float(), requires_grad=False) * terminal_gen_action_prob[:, 0] * vocab_tgt_prob + \
                   Var(tgt_action_seq_type[:, 2].float(), requires_grad=False) * terminal_gen_action_prob[:, 1] * copy_tgt_prob

        # nll loss
        loss = torch.neg(torch.sum(tgt_prob))
        return loss





