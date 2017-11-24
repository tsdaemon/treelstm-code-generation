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
from model.decoder import Hyp

sys.setrecursionlimit(50000)


class Tree2TreeModel(nn.Module):
    def __init__(self, config, word_embeds, terminal_vocab, grammar):
        super().__init__()

        self.config = config
        self.cuda = config.cuda

        self.terminal_vocab = terminal_vocab
        self.grammar = grammar

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
        self.softmax = nn.Softmax()

    def forward(self, tree, query_tokens, query):
        vocab_embedding = self.vocab_embedding_W.data
        rule_embedding = self.rule_embedding_W.data

        h, c, query_embed = self.forward_encode(tree, query_tokens)

        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp(self.grammar)
        root_hyp.state = zeros(self.config.decoder_hidden_dim, self.cuda)
        root_hyp.cell = zeros(self.config.decoder_hidden_dim, self.cuda)
        root_hyp.action_embed = zeros(self.config.rule_embed_dim, self.cuda)
        root_hyp.node_id = self.grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]  # [list() for i in range(live_hyp_num)]

        # source word id in the terminal vocab
        src_token_id = [self.terminal_vocab.convertToIdx(t, Constants.UNK) for t in query][
                       :self.config.max_query_length]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == Constants.UNK]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else:
                token_set.add(tid)

        for t in range(self.config.max_time_step):
            hyp_num = len(hyp_samples)
            # print 'time step [%d]' % t
            decoder_prev_state = from_list([hyp.state for hyp in hyp_samples], self.cuda)
            decoder_prev_cell = from_list([hyp.cell for hyp in hyp_samples], self.cuda)

            hist_h = zeros(hyp_num, self.config.max_time_step, self.config.decoder_hidden_dim, cuda=self.cuda)

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = hyp.hist_h
                    # for j, h in enumerate(hyp.hist_h):
                    #    hist_h[i, j] = h

            prev_action_embed = from_list([hyp.action_embed for hyp in hyp_samples], self.cuda)
            node_id = from_list([hyp.node_id for hyp in hyp_samples], self.cuda)
            parent_rule_id = from_list([hyp.parent_rule_id for hyp in hyp_samples], self.cuda)
            parent_t = from_list([hyp.get_action_parent_t() for hyp in hyp_samples], self.cuda)
            query_embed_tiled = query_embed.repeat([live_hyp_num, 1, 1])

            inputs = [np.array([t], dtype='int32'), decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, \
            rule_prob, gen_action_prob, vocab_prob, copy_prob = self.decoder_func_next_step(*inputs)

            new_hyp_samples = []

            cut_off_k = beam_size
            score_heap = []

            # iterating over items in the beam
            # print 'time step: %d, hyp num: %d' % (t, live_hyp_num)

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = np.array([hyp.score for hyp in hyp_samples])

            # word_prob[:, src_token_id] += gen_action_prob[:, 1:2] * copy_prob[:, :len(src_token_id)]
            # word_prob[:, unk] = 0

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in xrange(live_hyp_num):
                hyp = hyp_samples[k]

                # if k == 0:
                #     print 'Top Hyp: %s' % hyp.tree.__repr__()

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                assert hyp, 'none hyp!'

                # if it's not a leaf
                if not grammar.is_value_node(frontier_nt):
                    # iterate over all the possible rules
                    rules = grammar[frontier_nt.as_type_node] if config.head_nt_constraint else grammar
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = np.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf that holds values
                    cand_copy_prob = 0.0
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]

                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score

                        unk_word = example.query[unk_pos]
                        unk_words.append(unk_word)

                        cand_copy_prob = gen_action_prob[k, 1]

                    word_gen_hyp_ids.append(k)
                    cand_copy_probs.append(cand_copy_prob)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = np.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids, None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            # expand_cand_num = 0
            for cand_id in top_cand_ids:
                # cand is rule application
                new_hyp = None
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp(hyp)
                    new_hyp.apply_rule(rule)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    # if frontier_nt.type == int and (not (is_numeric(token) or token == '<eos>')):
                    #     continue

                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]

                    new_hyp = Hyp(hyp)
                    new_hyp.append_token(token)

                    if log:
                        cand_copy_prob = cand_copy_probs[word_gen_hyp_id]
                        if cand_copy_prob > 0.5:
                            new_hyp.log += ' || ' + str(new_hyp.frontier_nt()) + '{copy[%s][p=%f]}' % (
                                token, cand_copy_prob)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]
                    new_hyp.node_id = grammar.get_node_type_id(frontier_nt)

                # get the new frontier nt after rule application
                new_frontier_nt = new_hyp.frontier_nt()

                # if new_frontier_nt is None, then we have a new completed hyp!
                if new_frontier_nt is None:
                    # if t <= 1:
                    #     continue

                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1

                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    # new_hyp.parent_rule_id = grammar.rule_to_id[
                    #     new_frontier_nt.parent.to_rule(include_value=False)]
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]

                    new_hyp_samples.append(new_hyp)

                    # expand_cand_num += 1
                    # if expand_cand_num >= beam_size - completed_hyp_num:
                    #     break

                    # cand is word generation

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples
            # hyp_samples = sorted(new_hyp_samples, key=lambda x: x.score, reverse=True)[:live_hyp_num]

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return completed_hyps

    def forward_encode(self, tree, query):
        # get encoder results
        # (max_query_length, word_embed_dim)
        query_embed = Var(self.word_embedding[query], requires_grad=False)
        # (decoder_hidden_dim), (max_query_length, decoder_hidden_dim)
        return self.encoder(tree, query_embed)

    def forward_decoder_step(self, t, h, c, prev_action_embed, hist_h, ctx, node_id, par_rule_id, parent_h):
        # (node_embed_dim)
        node_embed = self.node_embedding[node_id]

        # (decoder_state_dim)
        if par_rule_id < 0:
            par_rule_embed = zeros(1, self.config.rule_embed_dim, cuda=self.cuda)
        else:
            par_rule_embed = self.rule_embedding_W[par_rule_id]

        # (1, decoder_state_dim)
        prev_action_embed_reshaped = prev_action_embed.unsqueeze(1)

        # (1, node_embed_dim)
        node_embed_reshaped = node_embed.unsqueeze(1)

        # (1, node_embed_dim)
        par_rule_embed_reshaped = par_rule_embed.unsqueeze(1)

        if not self.config.frontier_node_type_feed:
            node_embed_reshaped *= 0.

        if not self.config.parent_action_feed:
            par_rule_embed_reshaped *= 0.

        decoder_input = torch.cat([prev_action_embed_reshaped, node_embed_reshaped, par_rule_embed_reshaped], dim=-1)

        h, c, ctx_vec = self.decoder(t,
                                     decoder_input,
                                     ctx,
                                     hist_h,
                                     h,
                                     c,
                                     parent_h)

        # (decoder_hidden_state + encoder_hidden_dim)
        decoder_concat = torch.cat([h, ctx_vec], dim=-1)

        # (rule_embed_dim)
        decoder_hidden_state_trans_rule = self.decoder_hidden_state_W_rule(h)
        decoder_hidden_state_trans_token = self.decoder_hidden_state_W_token(decoder_concat)

        # (rule_num)
        rule_predict = self.softmax(
            torch.addmm(self.rule_embedding_b, decoder_hidden_state_trans_rule, torch.t(self.rule_embedding_W)))

        # (2)
        terminal_gen_action_prob = self.terminal_gen_softmax(h)

        # (target_vocab_size)
        vocab_predict = self.softmax(
            torch.addmm(self.vocab_embedding_b, decoder_hidden_state_trans_token, torch.t(self.vocab_embedding_W)))

        # (max_query_length)
        copy_prob = self.src_ptr_net(ctx, decoder_concat)


    def forward_train(self, tree, query, tgt_node_seq, tgt_action_seq, tgt_par_rule_seq, tgt_par_t_seq,
                      tgt_action_seq_type):
        # encoder
        h, c, ctx = self.forward_encode(tree, query)

        # prepare output for teacher forced decoding
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

        # (max_example_action_num, rule_embed_dim + node_embed_dim + rule_embed_dim)
        decoder_input = torch.cat([tgt_action_seq_embed_tm1, tgt_node_embed, tgt_par_rule_embed], dim=-1)

        # (max_example_action_num, decoder_hidden_dim), (max_example_action_num, encoder_hidden_dim)
        decoder_hidden_states, ctx_vectors = self.decoder.forward_train(decoder_input,
                                                                        ctx, h, c, tgt_par_t_seq)

        # (max_example_action_num, decoder_hidden_state + encoder_hidden_dim)
        decoder_concat = torch.cat([decoder_hidden_states, ctx_vectors], dim=-1)

        # (max_example_action_num, rule_embed_dim)
        decoder_hidden_state_trans_rule = self.decoder_hidden_state_W_rule(decoder_hidden_states)
        decoder_hidden_state_trans_token = self.decoder_hidden_state_W_token(decoder_concat)

        # (max_example_action_num, rule_num)
        rule_predict = self.log_softmax(
            torch.addmm(self.rule_embedding_b, decoder_hidden_state_trans_rule, torch.t(self.rule_embedding_W)))

        # (max_example_action_num, 2)
        terminal_gen_action_prob = self.terminal_gen_softmax(decoder_hidden_states)

        # (max_example_action_num, target_vocab_size)
        vocab_predict = self.log_softmax(
            torch.addmm(self.vocab_embedding_b, decoder_hidden_state_trans_token, torch.t(self.vocab_embedding_W)))

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
                   Var(tgt_action_seq_type[:, 1].float(), requires_grad=False) * terminal_gen_action_prob[:,
                                                                                 0] * vocab_tgt_prob + \
                   Var(tgt_action_seq_type[:, 2].float(), requires_grad=False) * terminal_gen_action_prob[:,
                                                                                 1] * copy_tgt_prob

        # nll loss
        loss = torch.neg(torch.sum(tgt_prob))
        return loss
