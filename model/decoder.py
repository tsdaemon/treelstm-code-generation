import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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
        init.uniform(self.wi.weight)
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
        init.uniform(self.wf.weight)
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
        init.uniform(self.wc.weight)
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
        init.uniform(self.wo.weight)
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
        init.uniform(self.att_ctx.weight)
        self.att_h = nn.Linear(output_dim, att_hidden_dim, bias=False)
        init.uniform(self.att_h.weight)
        self.att = nn.Linear(att_hidden_dim, 1)
        init.uniform(self.att.weight)

        # attention over history
        self.h_att_hist = nn.Linear(output_dim, att_hidden_dim)
        init.uniform(self.h_att_hist.weight)
        self.h_att_h = nn.Linear(output_dim, att_hidden_dim, bias=False)
        init.uniform(self.h_att_h.weight)
        self.h_att = nn.Linear(att_hidden_dim, 1)
        init.uniform(self.h_att.weight)

        self.dropout = nn.AlphaDropout(p=p_dropout)
        self.cuda = config.cuda
        self.parent_hidden_state_feed = config.parent_hidden_state_feed

    def forward(self, t, X, context, hist_h, h, c, parent_t):
        # (input_dim)
        X = self.dropout(X)

        # (output_dim)
        xi = self.wi(X)
        xf = self.wf(X)
        xc = self.wc(X)
        xo = self.wo(X)

        # (context_size, att_layer1_dim)
        context_att_trans = self.att_ctx(context)

        # if not init_state:
        #     init_state = torch.cuda.FloatTensor(self.output_dim) if self.cuda else torch.FloatTensor(self.output_dim)
        #     init_state = init_state.zero_()
        #
        # if not init_cell:
        #     init_cell = torch.cuda.FloatTensor(self.output_dim) if self.cuda else torch.FloatTensor(self.output_dim)
        #     init_cell = init_cell.zero_()

        # (att_layer1_dim)
        h_att_trans = self.att_h(h)

        # (context_size, att_layer1_dim)
        att_hidden = F.tanh(context_att_trans + h_att_trans.unsqueeze(0))
        # (context_size, 1)
        att_raw = self.att(att_hidden)
        # att_raw = att_raw.reshape((att_raw.shape[0], att_raw.shape[1]))

        # (context_size)
        ctx_att = F.exp(att_raw - att_raw.max(dim=-1, keepdim=True))

        # if context_mask:
        #     ctx_att = ctx_att * context_mask

        ctx_att = ctx_att / ctx_att.sum(dim=-1, keepdim=True)
        # (context_dim)
        ctx_vec = (context * ctx_att.unsqueeze(1)).sum(axis=1)

        ##### attention over history #####
        def _attention_over_history():
            # hist_h_mask = torch.CharTensor(hist_h.shape).zero_()
            # hist_h_mask.put() = T.set_subtensor(hist_h_mask[:, T.arange(t)], 1)

            # hist_h - (seq_len, output_dim)
            # (seq_len, att_hidden_dim)
            hist_h_att_trans = self.h_att_hist(hist_h)

            # h - (output_dim)
            # (att_hidden_dim)
            h_hatt_trans = self.h_att_h(h)
            # (seq_len, att_hidden_dim)
            hatt_hidden = F.tanh(hist_h_att_trans + h_hatt_trans.unsqueeze(0))
            # (seq_len, 1)
            hatt_raw = self.h_att(hatt_hidden)
            # hatt_raw = hatt_raw.reshape((hist_h.shape[0], hist_h.shape[1]))

            # (seq_len, 1)
            hatt_exp = F.exp(hatt_raw - torch.max(hatt_raw, dim=-1, keepdim=True)) # * hist_h_mask
            # (seq_len, 1)
            h_att_weights = hatt_exp / (torch.sum(hatt_exp, dim=-1, keepdim=True) + 1e-7)

            # (output_dim)
            _h_ctx_vec = torch.sum(hist_h * h_att_weights, axis=1)

            return _h_ctx_vec

        if t:
            h_ctx_vec = _attention_over_history()
        else:
            h_ctx_vec = torch.zeros(h.shape)

        ##### feed in parent hidden state #####
        if t and self.parent_hidden_state_feed:
            par_h = hist_h[parent_t, :]
        else:
            par_h = torch.zeros(h.shape)

        i = F.sigmoid(self.wi(X) + self.ui(h) + self.ci(ctx_vec) + self.pi(par_h) + self.hi(h_ctx_vec))
        f = F.sigmoid(self.wf(X) + self.uf(h) + self.cf(ctx_vec) + self.pf(par_h) + self.hf(h_ctx_vec))
        c_new = f * c + i * F.tanh(self.wc(X) + self.uc(h) + self.cc(ctx_vec) +
                                   self.pc(par_h) + self.hc(h_ctx_vec))
        o = F.sigmoid(self.wo(X) + self.uo(h) + self.co(ctx_vec) + self.po(par_h) + self.ho(h_ctx_vec))
        # else:
        #     i_t = self.inner_activation(
        #         xi_t + T.dot(h_tm1 * b_u[0], u_i) + T.dot(ctx_vec, c_i) + T.dot(par_h, p_i))  # + T.dot(h_ctx_vec, h_i)
        #     f_t = self.inner_activation(
        #         xf_t + T.dot(h_tm1 * b_u[1], u_f) + T.dot(ctx_vec, c_f) + T.dot(par_h, p_f))  # + T.dot(h_ctx_vec, h_f)
        #     c_t = f_t * c_tm1 + i_t * self.activation(
        #         xc_t + T.dot(h_tm1 * b_u[2], u_c) + T.dot(ctx_vec, c_c) + T.dot(par_h, p_c))  # + T.dot(h_ctx_vec, h_c)
        #     o_t = self.inner_activation(
        #         xo_t + T.dot(h_tm1 * b_u[3], u_o) + T.dot(ctx_vec, c_o) + T.dot(par_h, p_o))  # + T.dot(h_ctx_vec, h_o)

        h = o * self.activation(c_new)

        # h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        # c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        hist_h[parent_t, :] = h

        return h, c, ctx_vec, hist_h

        outputs = outputs.dimshuffle((1, 0, 2))
        ctx_vectors = ctx_vectors.dimshuffle((1, 0, 2))
        cells = cells.dimshuffle((1, 0, 2))

        return outputs, cells, ctx_vectors


class PointerNet(nn.Module):
    def __init__(self, config):
        super(PointerNet, self).__init__()

        self.dense1_input = nn.Linear(config.encoder_hidden_dim, config.ptrnet_hidden_dim)
        init.uniform(self.dense1_input.weight)

        self.dense1_h = nn.Linear(config.decoder_hidden_dim + config.encoder_hidden_dim, config.ptrnet_hidden_dim)
        init.uniform(self.dense1_h.weight)

        self.dense2 = nn.Linear(config.ptrnet_hidden_dim, 1)
        init.uniform(self.dense2.weight)

    def forward(self, query_embed, decoder_states):
        query_embed_trans = self.dense1_input(query_embed)
        h_trans = self.dense1_h(decoder_states)

        # (max_decode_step, query_token_num, ptr_net_hidden_dim)
        dense1_trans = F.tanh(query_embed_trans + h_trans)

        scores = self.dense2(dense1_trans).flatten(3)

        scores = F.exp(scores - F.max(scores, dim=-1, keepdim=True))
        scores = scores / F.sum(scores, dim=-1, keepdim=True)

        return scores
