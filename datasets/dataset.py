from copy import deepcopy
import torch.utils.data as data
import torch
import os
import logging

import Constants
from natural_lang.tree import *
from utils.io import deserialize_from_file
from lang.action import *
from lang.parse import *

parents_prefix = {
    'ccg': 'ccg',
    'pcfg': 'constituency',
    'dependency': 'dependency'
}


class Dataset(data.Dataset):
    def __init__(self, data_dir, file_name, grammar, vocab, terminal_vocab, config):
        super(Dataset, self).__init__()
        self.vocab = vocab
        self.terminal_vocab = terminal_vocab
        self.grammar = grammar

        self.config = config

        self.load_queries(data_dir, file_name, config.syntax)
        self.size = self.load_output(data_dir, file_name)
        self.init_actions_tensors()

    def prepare_torch(self, cuda):
        if cuda:
            self.queries = [q.cuda() for q in self.queries]
            self.tgt_node_seq = [q.cuda() for q in self.tgt_node_seq]
            self.tgt_par_rule_seq = [q.cuda() for q in self.tgt_par_rule_seq]
            self.tgt_par_t_seq = [q.cuda() for q in self.tgt_par_t_seq]
            self.tgt_action_seq = [q.cuda() for q in self.tgt_action_seq]
            self.tgt_action_seq_type = [q.cuda() for q in self.tgt_action_seq_type]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        enc_tree = deepcopy(self.query_trees[index])

        query = self.queries[index]
        query_tokens = self.query_tokens[index]
        query_tokens = self.fix_seq_length_one(query_tokens, query.size(), Constants.PAD)

        tgt_node_seq = self.tgt_node_seq[index]
        tgt_par_rule_seq = self.tgt_par_rule_seq[index]
        tgt_par_t_seq = self.tgt_par_t_seq[index]
        tgt_action_seq = self.tgt_action_seq[index]
        tgt_action_seq_type = self.tgt_action_seq_type[index]

        code = deepcopy(self.codes[index])
        code_tree = deepcopy(self.code_trees[index])

        return enc_tree, query, query_tokens, \
               tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq, tgt_action_seq, tgt_action_seq_type, \
               code, code_tree

    def get_batch(self, indices):
        trees = [deepcopy(self.query_trees[index]) for index in indices]
        max_tree_length = max([tree.size() for tree in trees])

        queries = self.get_seq_batch(self.queries, indices, max_tree_length, Constants.PAD)

        max_out_length = min(max([len(self.tgt_node_seq[index]) for index in indices]), self.config.max_actions_length)

        tgt_node_seq = self.get_seq_batch(self.tgt_node_seq, indices, max_out_length, 0)
        tgt_par_rule_seq = self.get_seq_batch(self.tgt_par_rule_seq, indices, max_out_length, 0)
        tgt_par_t_seq = self.get_seq_batch(self.tgt_par_t_seq, indices, max_out_length, 0)
        tgt_action_seq = self.get_seq_batch(self.tgt_action_seq, indices, max_out_length, 0)
        tgt_action_seq_type = self.get_seq_batch(self.tgt_action_seq_type, indices, max_out_length, 0)

        return trees, queries, \
               tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq, \
               tgt_action_seq, tgt_action_seq_type

    def get_seq_batch(self, seq, indices, max_size, pad_item):
        seq = [seq[index] for index in indices]
        seq = self.fix_seq_length(seq, max_size, pad_item)
        return torch.stack(seq)

    def load_queries(self, data_dir, file_name, syntax):
        parents_file = os.path.join(data_dir, '{}.in.{}_parents'.format(file_name, parents_prefix[syntax]))
        tokens_file = os.path.join(data_dir, '{}.in.tokens'.format(file_name))

        logging.info('Reading query trees...')
        self.query_trees = self.read_query_trees(parents_file)

        logging.info('Reading query tokens...')
        self.queries, self.query_tokens = self.read_query(tokens_file)
        # self.queries = self.fix_query_length(self.queries)
        # self.queries = torch.stack(self.queries)

    def read_query(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            query_and_tokens = [self.read_query_line(line) for line in tqdm(f.readlines())]
        # unzip
        return tuple(zip(*query_and_tokens))

    def read_query_line(self, line):
        tokens = line.split()
        indices = self.vocab.convertToIdx(tokens, Constants.UNK_WORD)
        return torch.LongTensor(indices), tokens

    def fix_seq_length(self, seqns, max_size, pad_item):
        ls = []
        for seq in seqns:
            seq_padded = self.fix_seq_length_one(seq, max_size, pad_item)
            ls.append(seq_padded)
        return ls

    def fix_seq_length_one(self, seq, max_size, pad_item):
        if len(seq) < max_size:
            size_0 = max_size-len(seq)
            size_next = list(seq.size()[1:])
            size_pad = [size_0] + size_next
            pads = torch.LongTensor(*size_pad)
            if seq.is_cuda:
                pads = pads.cuda()
            pads = pads.fill_(pad_item)

            return torch.cat([seq, pads], dim=0)
        else:
            return seq[:max_size]

    def read_query_trees(self, filename):
        with open(filename, 'r') as f:
            trees = list(map(read_tree, tqdm(f.readlines())))
        return trees

    def load_output(self, data_dir, file_name):
        logging.info('Reading code files...')
        if self.config.unary_closures:
            trees_file = '{}.out.trees.uc.bin'.format(file_name)
        else:
            trees_file = '{}.out.trees.bin'.format(file_name)

        trees_file = os.path.join(data_dir, trees_file)
        code_file = os.path.join(data_dir, '{}.out.bin'.format(file_name))
        self.code_trees = deserialize_from_file(trees_file)
        self.codes = deserialize_from_file(code_file)

        logging.info('Constructing code representation...')
        self.actions = []

        for code_tree, query_tokens in tqdm(zip(self.code_trees, self.query_tokens)):
            rule_list, rule_parents = code_tree.get_productions(include_value_node=True)

            actions = []
            rule_pos_map = dict()

            for rule_count, rule in enumerate(rule_list):
                if not self.grammar.is_value_node(rule.parent):
                    assert rule.value is None
                    parent_rule = rule_parents[(rule_count, rule)][0]
                    if parent_rule:
                        parent_t = rule_pos_map[parent_rule]
                    else:
                        parent_t = 0

                    rule_pos_map[rule] = len(actions)

                    d = {'rule': rule, 'parent_t': parent_t, 'parent_rule': parent_rule}
                    action = Action(APPLY_RULE, d)

                    actions.append(action)
                else:
                    assert rule.is_leaf

                    parent_rule = rule_parents[(rule_count, rule)][0]
                    parent_t = rule_pos_map[parent_rule]

                    terminal_val = rule.value
                    terminal_str = str(terminal_val)
                    terminal_tokens = get_terminal_tokens(terminal_str)

                    for terminal_token in terminal_tokens:
                        term_tok_id = self.terminal_vocab.getIndex(terminal_token, Constants.UNK)
                        tok_src_idx = -1
                        try:
                            tok_src_idx = query_tokens.index(terminal_token)
                        except ValueError:
                            pass

                        d = {'literal': terminal_token, 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}

                        # cannot copy, only generation
                        # could be unk!
                        if tok_src_idx < 0:
                            action = Action(GEN_TOKEN, d)
                        else:  # copy
                            if term_tok_id != Constants.UNK:
                                d['source_idx'] = tok_src_idx
                                action = Action(GEN_COPY_TOKEN, d)
                            else:
                                d['source_idx'] = tok_src_idx
                                action = Action(COPY_TOKEN, d)

                        actions.append(action)

                    d = {'literal': '<eos>', 'rule': rule, 'parent_rule': parent_rule, 'parent_t': parent_t}
                    actions.append(Action(GEN_TOKEN, d))

            if len(actions) == 0:
                continue

            self.actions.append(actions)
        return len(self.actions)

    def init_actions_tensors(self):
        terminal_vocab = self.terminal_vocab

        logging.info('Initializing actions...')
        self.tgt_node_seq = []
        self.tgt_par_rule_seq = []
        self.tgt_par_t_seq = []
        self.tgt_action_seq = []
        self.tgt_action_seq_type = []

        for actions in self.actions:
            size = len(actions)
            tgt_node_seq = torch.LongTensor(size).zero_()
            tgt_par_rule_seq = torch.LongTensor(size).zero_()
            tgt_par_t_seq = torch.LongTensor(size).zero_()
            tgt_action_seq = torch.LongTensor(size, 3).zero_()
            tgt_action_seq_type = torch.LongTensor(size, 3).zero_()

            for t, action in enumerate(actions):
                if action.act_type == APPLY_RULE:
                    rule = action.data['rule']
                    tgt_action_seq[t, 0] = self.grammar.rule_to_id[rule]
                    tgt_action_seq_type[t, 0] = 1
                elif action.act_type == GEN_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab.getIndex(token, Constants.UNK)
                    tgt_action_seq[t, 1] = token_id
                    tgt_action_seq_type[t, 1] = 1
                elif action.act_type == COPY_TOKEN:
                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[t, 2] = src_token_idx
                    tgt_action_seq_type[t, 2] = 1
                elif action.act_type == GEN_COPY_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab.getIndex(token, Constants.UNK)
                    tgt_action_seq[t, 1] = token_id
                    tgt_action_seq_type[t, 1] = 1

                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[t, 2] = src_token_idx
                    tgt_action_seq_type[t, 2] = 1
                else:
                    raise RuntimeError('wrong action type!')

                # parent information
                rule = action.data['rule']
                parent_rule = action.data['parent_rule']
                tgt_node_seq[t] = self.grammar.get_node_type_id(rule.parent)
                if parent_rule:
                    tgt_par_rule_seq[t] = self.grammar.rule_to_id[parent_rule]
                else:
                    assert t == 0
                    tgt_par_rule_seq[t] = -1

                # parent hidden states
                parent_t = action.data['parent_t']
                tgt_par_t_seq[t] = parent_t

            self.tgt_node_seq.append(tgt_node_seq)
            self.tgt_par_rule_seq.append(tgt_par_rule_seq)
            self.tgt_par_t_seq.append(tgt_par_t_seq)
            self.tgt_action_seq.append(tgt_action_seq)
            self.tgt_action_seq_type.append(tgt_action_seq_type)
