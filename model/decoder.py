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
                 config,
                 p_dropout=0.0):

        super(CondAttLSTM, self).__init__()

        self.output_dim = output_dim
        self.context_dim = context_dim
        self.input_dim = input_dim

        # input gate
        self.wi = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.wi.weight)
        self.ui = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.ui.weight)
        self.ci = nn.Linear(context_dim, output_dim, bias=False)
        init.orthogonal(self.ci.weight)
        self.hi = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.hi.weight)
        self.pi = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.pi.weight)

        # forget gate
        self.wf = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.wf.weight)
        self.uf = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.uf.weight)
        self.cf = nn.Linear(context_dim, output_dim, bias=False)
        init.orthogonal(self.cf.weight)
        self.hf = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.hf.weight)
        self.pf = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.pf.weight)

        # memory cell new value
        self.wc = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.wc.weight)
        self.uc = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.uc.weight)
        self.cc = nn.Linear(context_dim, output_dim, bias=False)
        init.orthogonal(self.cc.weight)
        self.hc = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.hc.weight)
        self.pc = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.pc.weight)

        # output gate
        self.wo = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.wo.weight)
        self.uo = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.uo.weight)
        self.co = nn.Linear(context_dim, output_dim, bias=False)
        init.orthogonal(self.co.weight)
        self.ho = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.ho.weight)
        self.po = nn.Linear(output_dim, output_dim, bias=False)
        init.orthogonal(self.po.weight)

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

        self.dropout = nn.AlphaDropout(p=p_dropout)
        self.cuda = config.cuda
        self.parent_hidden_state_feed = config.parent_hidden_state_feed

        self.softmax = nn.Softmax();

    def forward(self, t, X, context, hist_h, h, c, parent_h):
        # (input_dim)
        X = self.dropout(X)
        return self.forward_node(t, X[t], context, hist_h, h, c, parent_h)

    def forward_node(self, t, X, context, hist_h, h, c, par_h):
        # (context_size, att_layer1_dim)
        context_att_trans = self.att_ctx(context)

        # (att_layer1_dim)
        h_att_trans = self.att_h(h)

        # (context_size, att_layer1_dim)
        att_hidden = F.tanh(context_att_trans + h_att_trans.unsqueeze(0))

        # (1, context_size)
        att_raw = self.att(att_hidden).view(1, -1)
        # att_raw = att_raw.reshape((att_raw.shape[0], att_raw.shape[1]))

        # (context_size, 1)
        ctx_att = self.softmax(att_raw).view(-1, 1)

        # (context_dim)
        ctx_vec = (context * ctx_att).sum(dim=0)

        def _attention_over_history():
            # hist_h - (seq_len, output_dim)
            # (seq_len, att_hidden_dim)
            hist_h_att_trans = self.h_att_hist(hist_h)

            # h - (output_dim)
            # (att_hidden_dim)
            h_hatt_trans = self.h_att_h(h)
            # (seq_len, att_hidden_dim)
            hatt_hidden = F.tanh(hist_h_att_trans + h_hatt_trans.unsqueeze(0))
            # (seq_len, 1)
            hatt_raw = self.h_att(hatt_hidden).view(1, -1)
            # hatt_raw = hatt_raw.reshape((hist_h.shape[0], hist_h.shape[1]))

            # (seq_len, 1)
            h_att_weights = self.softmax(hatt_raw).view(-1, 1)

            # (output_dim)
            _h_ctx_vec = torch.sum(hist_h * h_att_weights, dim=0)

            return _h_ctx_vec

        if t:
            h_ctx_vec = _attention_over_history()
        else:
            h_ctx_vec = Var(zeros_like(h, self.cuda))

        i = F.sigmoid(self.wi(X) + self.ui(h) + self.ci(ctx_vec) + self.pi(par_h) + self.hi(h_ctx_vec))
        f = F.sigmoid(self.wf(X) + self.uf(h) + self.cf(ctx_vec) + self.pf(par_h) + self.hf(h_ctx_vec))
        c_new = f * c + i * F.tanh(self.wc(X) + self.uc(h) + self.cc(ctx_vec) +
                                   self.pc(par_h) + self.hc(h_ctx_vec))
        o = F.sigmoid(self.wo(X) + self.uo(h) + self.co(ctx_vec) + self.po(par_h) + self.ho(h_ctx_vec))

        h = o * F.tanh(c_new)

        return h, c, ctx_vec

    # Teacher forcing: Feed the target as the next input
    def forward_train(self, X, context, h, c, parent_t):
        length = len(X)
        # (max_sequence_length, input_dim)
        X = self.dropout(X)
        # (max_sequence_length, decoder_hidden_dim)
        output_h = None
        # (max_sequence_length, context_size)
        output_ctx = []

        for t in range(length):
            # extract parent node from history
            if t and self.parent_hidden_state_feed:
                par_h = output_h[parent_t[t], :]
            else:
                par_h = Var(zeros_like(h, self.cuda))

            h, c, ctx_vec = self.forward_node(t, X[t], context, output_h, h, c, par_h)
            if output_h is None:
                output_h = h.unsqueeze(0)
            else:
                output_h = torch.cat([output_h, h.unsqueeze(0)])
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

        self.log_softmax = nn.LogSoftmax()

    def forward(self, ctx, decoder_states):
        ctx_trans = self.dense1_input(ctx)
        decoder_trans = self.dense1_h(decoder_states)

        ctx_trans = ctx_trans.unsqueeze(0)
        decoder_trans = decoder_trans.unsqueeze(1)

        # (max_decode_step, query_token_num, ptr_net_hidden_dim)
        dense1_trans = F.tanh(ctx_trans + decoder_trans)

        scores = self.dense2(dense1_trans).squeeze(2)

        scores = self.log_softmax(scores)

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
            self.t=-1
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
