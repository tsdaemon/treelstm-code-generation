import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable as Var

from model.utils import add_padding_and_stack


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, p_dropout=0.0):
        super().__init__()

        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        init.xavier_uniform(self.ioux.weight)
        self.ioux.bias = nn.Parameter(torch.FloatTensor(3*self.mem_dim).zero_())

        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        init.orthogonal(self.iouh.weight)

        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        init.xavier_uniform(self.fx.weight)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        init.orthogonal(self.fh.weight)

        self.fb = nn.Parameter(torch.FloatTensor(self.mem_dim).fill_(1.0))

        # self.init_h = nn.Parameter(torch.FloatTensor(1, self.mem_dim).zero_())
        # self.init_c = nn.Parameter(torch.FloatTensor(1, self.mem_dim).zero_())

        self.dropout = nn.AlphaDropout(p=p_dropout)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        # u is c tilda - the new value of memory cell
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1) +
            self.fb
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward_inner(self, tree, inputs):
        _ = [self.forward_inner(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            # child_c = self.init_c
            # child_h = self.init_h
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state

    def forward(self, tree, inputs):
        inputs = self.dropout(inputs)
        self.forward_inner(tree, inputs)
        return tree.state[1].squeeze(), tree.state[0].squeeze(), torch.stack([t.state[1] for t in tree.data()]).squeeze()


class EncoderLSTMWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if self.config.encoder == 'recursive-lstm':
            self.encoder = ChildSumTreeLSTM(config.word_embed_dim, config.encoder_hidden_dim, config.dropout)
        elif self.config.encoder == 'bi-lstm':
            hidden_dim = int(config.encoder_hidden_dim/2)
            # http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
            self.encoder = nn.LSTM(config.word_embed_dim, hidden_dim, 1, batch_first=True, dropout=config.dropout, bidirectional=True)
            self.init_bilstm(hidden_dim)

            self.init_h = nn.Parameter(torch.FloatTensor(2, 1, hidden_dim).zero_())
            self.init_c = nn.Parameter(torch.FloatTensor(2, 1, hidden_dim).zero_())
        else:
            raise Exception("Unknown encoder type!")

    def init_bilstm(self, hidden_dim):
        init.xavier_uniform(self.encoder.weight_ih_l0)
        init.xavier_uniform(self.encoder.weight_ih_l0_reverse)
        init.orthogonal(self.encoder.weight_hh_l0)
        init.orthogonal(self.encoder.weight_hh_l0_reverse)

        bias = self.init_lstm_bias(self.encoder.bias_ih_l0, hidden_dim)
        self.encoder.bias_ih_l0 = nn.Parameter(bias.clone())
        self.encoder.bias_hh_l0 = nn.Parameter(bias.clone())
        self.encoder.bias_ih_l0_reverse = nn.Parameter(bias.clone())
        self.encoder.bias_hh_l0_reverse = nn.Parameter(bias.clone())

    def init_lstm_bias(self, bias, hidden_dim):
        bias = bias.data.fill_(0.0)
        # forget gate
        bias[hidden_dim:2 * hidden_dim] = 1.0
        return bias

    def forward(self, trees, inputs):
        if self.config.encoder == 'recursive-lstm':
            return self.forward_recursive(trees, inputs)
        elif self.config.encoder == 'bi-lstm':
            return self.forward_lstm(inputs)

    def forward_lstm(self, inputs):
        # (1, batch_size, encoder_hidden_dim)
        h0 = self.init_h.repeat(1, inputs.data.shape[0], 1)
        c0 = self.init_c.repeat(1, inputs.data.shape[0], 1)
        ctx, hc = self.encoder(inputs, (h0, c0))
        assert ctx.data.shape[0] == inputs.data.shape[0]
        assert ctx.data.shape[2] == self.config.encoder_hidden_dim
        h, c = hc
        # (batch_size, encoder_hidden_dim)
        h = h.permute(1, 0, 2).contiguous().view(-1, self.config.encoder_hidden_dim)
        c = c.permute(1, 0, 2).contiguous().view(-1, self.config.encoder_hidden_dim)

        return h, c, ctx

    def forward_recursive(self, trees, inputs):
        # (batch_size, encoder_hidden_dim)
        h = []
        c = []
        # (batch_size, max_query_length, encoder_hidden_dim)
        ctx = []
        # encoder can process only one tree at the time
        for tree, input in zip(trees, inputs):
            h1, c1, ctx1 = self.encoder(tree, input)
            h.append(h1)
            c.append(c1)
            ctx.append(ctx1)

        # all ctx must be one length to be stacked
        ctx = add_padding_and_stack(ctx, inputs.is_cuda)
        h = torch.stack(h)
        c = torch.stack(h)

        return h, c, ctx

