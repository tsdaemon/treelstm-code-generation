import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from model.utils import *
from lang.grammar import Grammar
from lang.astnode import DecodeTree


class CondAttLSTM(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 context_dim,
                 att_hidden_dim,
                 config):

        super(CondAttLSTM, self).__init__()

        self.output_dim = output_dim
        self.context_dim = context_dim
        self.input_dim = input_dim

        # input gate
        self.W_ix = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.i_x.weight)

        self.W_i = nn.Linear(output_dim + context_dim + output_dim + output_dim, output_dim, bias=False)
        init.orthogonal(self.i.weight)

        # forget gate
        self.W_fx = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.W_fx.weight)

        self.W_f = nn.Linear(output_dim + context_dim + output_dim + output_dim, output_dim, bias=False)
        init.orthogonal(self.W_f.weight)

        # memory cell new value
        self.W_cx = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.W_cx.weight)

        self.W_c = nn.Linear(output_dim + context_dim + output_dim + output_dim, output_dim, bias=False)
        init.orthogonal(self.W_c.weight)

        # output gate
        self.W_ox = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.W_ox.weight)

        self.W_o = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.W_o.weight)

        # attention layer
        self.att_ctx = nn.Linear(context_dim, att_hidden_dim)
        init.xavier_uniform(self.att_ctx.weight)
        self.att_h = nn.Linear(output_dim, att_hidden_dim, bias=False)
        init.xavier_uniform(self.att_h.weight)
        self.att = nn.Linear(att_hidden_dim, 1)
        init.xavier_uniform(self.att.weight)

        # attention over history
        self.h_att_hist = nn.Linear(output_dim, att_hidden_dim)
        init.xavier_uniform(self.h_att_hist.weight)
        self.h_att_h = nn.Linear(output_dim, att_hidden_dim, bias=False)
        init.xavier_uniform(self.h_att_h.weight)
        self.h_att = nn.Linear(att_hidden_dim, 1)
        init.xavier_uniform(self.h_att.weight)

        self.dropout = nn.AlphaDropout(p=config.dropout)
        self.softmax = nn.Softmax(dim=-1);

        self.cuda = config.cuda
        self.parent_hidden_state_feed = config.parent_hidden_state_feed

    # one time step at the time
    def forward(self, t, x, context, hist_h, h, c, parent_h):
        # (batch_size, input_dim)
        x = self.dropout(x)
        # (batch_size, output_dim)
        xi, xf, xo, xc = self.W_ix(x), self.W_fx(x), self.W_ox(x), self.W_cx(x)

        return self.forward_node(t,
                                 xi, xf, xo, xc,
                                 context, hist_h,
                                 h, c, parent_h)

    def forward_node(self, t,
                     xi, xf, xo, xc,
                     context, hist_h,
                     h, c, par_h):
        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = self.att_ctx(context)

        # (batch_size, att_layer1_dim)
        h_att_trans = self.att_h(h)

        # (batch_size, context_size, att_layer1_dim)
        att_hidden = F.tanh(context_att_trans + h_att_trans.unsqueeze(1))

        # (batch_size, context_size)
        att_raw = self.att(att_hidden)

        # (batch_size, context_size)
        ctx_att = self.softmax(att_raw)

        # (batch_size, context_dim)
        ctx_vec = (context * ctx_att).sum(dim=1)

        def _attention_over_history():
            # hist_h - (batch_size, seq_len, output_dim)
            # (batch_size, seq_len, att_hidden_dim)
            hist_h_att_trans = self.h_att_hist(hist_h)

            # h - (batch_size, output_dim)
            # (batch_size, att_hidden_dim)
            h_hatt_trans = self.h_att_h(h)

            # (batch_size, seq_len, att_hidden_dim)
            hatt_hidden = F.tanh(hist_h_att_trans + h_hatt_trans.unsqueeze(1))

            # (batch_size, seq_len)
            hatt_raw = self.h_att(hatt_hidden)

            # (batch_size, seq_len)
            h_att_weights = self.softmax(hatt_raw)

            # (batch_size, output_dim)
            _h_ctx_vec = torch.sum(hist_h * h_att_weights, dim=1)

            return _h_ctx_vec

        if t and self.config.tree_attention:
            h_ctx_vec = _attention_over_history()
        else:
            h_ctx_vec = Var(zeros_like(h, self.cuda))

        if not self.config.parent_hidden_state_feed:
            par_h *= 0.

        # (batch_size, output_dim + context_dim + output_dim + output_dim)
        h_comb = torch.cat([h, ctx_vec, par_h, h_ctx_vec], dim=-1)

        # (batch_size, output_dim)
        i = F.sigmoid(xi + self.W_i(h_comb))
        f = F.sigmoid(xf + self.W_f(h_comb))
        c = f * c + i * F.tanh(xc + self.W_c(h_comb))
        o = F.sigmoid(xo + self.W_o(h_comb))

        h = o * F.tanh(c)

        return h, c, ctx_vec

    # all inputs at the time (teacher forcing)
    def forward_train(self, X, context, h, c, parent_t):
        length = len(X.shape[1])
        # (batch_size, max_sequence_length, input_dim)
        X = self.dropout(X)
        # calculate all X dense transformation at once
        # (batch_size, max_sequence_length, output_dim)
        Xi, Xf, Xo, Xc = self.W_ix(X), self.W_fx(X), self.W_ox(X), self.W_cx(X)
        # (batch_size, max_sequence_length, decoder_hidden_dim)
        output_h = None
        # (batch_size, max_sequence_length, encoder_hidden_dim)
        output_ctx = []

        for t in range(length):
            # extract parent node from history
            if t and self.parent_hidden_state_feed:
                par_h = output_h[:, parent_t[t], :]
            else:
                par_h = Var(zeros_like(h, self.cuda))

            h, c, ctx_vec = \
                self.forward_node(t,
                                  Xi[:, t, :].sqeeze(1), Xf[:, t, :].sqeeze(1), Xo[:, t, :].sqeeze(1), Xc[:, t, :].sqeeze(1),
                                  context, output_h,
                                  h, c, par_h)
            if output_h is None:
                output_h = h.unsqueeze(1)
            else:
                output_h = torch.cat([output_h, h.unsqueeze(1)])
            output_ctx.append(ctx_vec)

        return output_h, torch.stack(output_ctx)


class PointerNet(nn.Module):
    def __init__(self, config):
        super(PointerNet, self).__init__()

        self.dense1_input = nn.Linear(config.encoder_hidden_dim, config.ptrnet_hidden_dim)
        init.xavier_uniform(self.dense1_input.weight)

        self.dense1_h = nn.Linear(config.decoder_hidden_dim + config.encoder_hidden_dim, config.ptrnet_hidden_dim)
        init.xavier_uniform(self.dense1_h.weight)

        self.dense2 = nn.Linear(config.ptrnet_hidden_dim, 1)
        init.xavier_uniform(self.dense2.weight)

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1);

    def forward_scores(self, ctx, decoder_states):
        # (batch_size, max_query_length, ptrnet_hidden_dim)
        ctx_trans = self.dense1_input(ctx)

        # (batch_size, max_decode_step, ptrnet_hidden_dim)
        decoder_trans = self.dense1_h(decoder_states)

        # (batch_size, 1, max_query_length, ptrnet_hidden_dim)
        ctx_trans = ctx_trans.unsqueeze(2)

        # (batch_size, max_decode_step, 1, ptrnet_hidden_dim)
        decoder_trans = decoder_trans.unsqueeze(2)

        # (batch_size, max_decode_step, max_query_length, ptr_net_hidden_dim)
        dense1_trans = F.tanh(ctx_trans + decoder_trans)

        # (batch_size,  max_decode_step, max_query_length, 1)
        scores = self.dense2(dense1_trans)

        # (batch_size,  max_decode_step, max_query_length)
        return scores.squeeze(3)

    def forward(self, ctx, decoder_states):
        # (batch_size,  max_decode_step, max_query_length)
        scores = self.forward_scores(ctx, decoder_states)

        # (batch_size,  max_decode_step, max_query_length)
        scores = self.softmax(scores)

        # (batch_size,  max_decode_step, max_query_length)
        return scores

    def forward_train(self, ctx, decoder_states):
        # (batch_size,  max_decode_step, max_query_length)
        scores = self.forward_scores(ctx, decoder_states)

        # (batch_size,  max_decode_step, max_query_length)
        scores = self.log_softmax(scores)

        # (batch_size,  max_decode_step, max_query_length)
        return scores


class Hyp:
    def __init__(self, *args):
        if isinstance(args[0], Hyp):
            hyp = args[0]
            self.grammar = hyp.grammar
            self.tree = hyp.tree.copy()
            self.t = hyp.t
            self.hist_h = list(hyp.hist_h)
            self.log = hyp.log
            self.has_grammar_error = hyp.has_grammar_error
        else:
            assert isinstance(args[0], Grammar)
            grammar = args[0]
            self.grammar = grammar
            self.tree = DecodeTree(grammar.root_node.type)
            self.t = -1
            self.hist_h = []
            self.log = ''
            self.has_grammar_error = False

        self.score = 0.0

        self.__frontier_nt = self.tree
        self.__frontier_nt_t = -1

    def __repr__(self):
        return self.tree.__repr__()

    def can_expand(self, node):
        if self.grammar.is_value_node(node):
            # if the node is finished
            if node.value is not None and node.value.endswith('<eos>'):
                return False
            return True
        elif self.grammar.is_terminal(node):
            return False

        return True

    def apply_rule(self, rule, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        # assert rule.parent.type == nt.type
        if rule.parent.type != nt.type:
            self.has_grammar_error = True

        self.t += 1
        # set the time step when the rule leading by this nt is applied
        nt.t = self.t
        # record the ApplyRule action that is used to expand the current node
        nt.applied_rule = rule

        for child_node in rule.children:
            child = DecodeTree(child_node.type, child_node.label, child_node.value)
            # if is_builtin_type(rule.parent.type):
            #     child.label = None
            #     child.holds_value = True

            nt.add_child(child)

    def append_token(self, token, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        self.t += 1

        if nt.value is None:
            # this terminal node is empty
            nt.t = self.t
            nt.value = token
        else:
            nt.value += token

    def frontier_nt_helper(self, node):
        if node.is_leaf:
            if self.can_expand(node):
                return node
            else:
                return None

        for child in node.children:
            result = self.frontier_nt_helper(child)
            if result:
                return result

        return None

    def frontier_nt(self):
        if self.__frontier_nt_t == self.t:
            return self.__frontier_nt
        else:
            _frontier_nt = self.frontier_nt_helper(self.tree)
            self.__frontier_nt = _frontier_nt
            self.__frontier_nt_t = self.t

            return _frontier_nt

    def get_action_parent_t(self):
        """
        get the time step when the parent of the current
        action was generated
        WARNING: 0 will be returned if parent if None
        """
        nt = self.frontier_nt()

        if nt.parent:
            return nt.parent.t
        else:
            return 0
