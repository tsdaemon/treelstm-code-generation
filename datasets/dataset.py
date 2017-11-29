from copy import deepcopy
import torch.utils.data as data
import torch
import os

import Constants
from natural_lang.tree import Tree
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
        self.init_data_matrices()

    def prepare_torch(self, cuda):
        if cuda:
            self.queries = self.queries.cuda()
            self.tgt_node_seq = self.tgt_node_seq.cuda()
            self.tgt_par_rule_seq = self.tgt_par_rule_seq.cuda()
            self.tgt_par_t_seq = self.tgt_par_t_seq.cuda()
            self.tgt_action_seq = self.tgt_action_seq.cuda()
            self.tgt_action_seq_type = self.tgt_action_seq_type.cuda()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        enc_tree = deepcopy(self.query_trees[index])

        query = self.queries[index]
        query_tokens = self.query_tokens[index]

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
        queries = self.queries[indices]

        tgt_node_seq = self.tgt_node_seq[indices]
        tgt_par_rule_seq = self.tgt_par_rule_seq[indices]
        tgt_par_t_seq = self.tgt_par_t_seq[indices]
        tgt_action_seq = self.tgt_action_seq[indices]
        tgt_action_seq_type = self.tgt_action_seq_type[indices]
        return trees, queries, \
               tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq, tgt_action_seq, tgt_action_seq_type

    def load_queries(self, data_dir, file_name, syntax):
        parents_file = os.path.join(data_dir, '{}.in.{}_parents'.format(file_name, parents_prefix[syntax]))
        tokens_file = os.path.join(data_dir, '{}.in.tokens'.format(file_name))

        print('Reading query trees...')
        self.query_trees = self.read_query_trees(parents_file)

        print('Reading query tokens...')
        self.queries, self.query_tokens = self.read_query(tokens_file)
        self.queries = self.fix_query_length(self.queries)
        self.queries = torch.stack(self.queries)

    def read_query(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            query_and_tokens = [self.read_query_line(line) for line in tqdm(f.readlines())]
        # unzip
        return tuple(zip(*query_and_tokens))

    def read_query_line(self, line):
        tokens = line.split()
        indices = self.vocab.convertToIdx(tokens, Constants.UNK_WORD)
        return torch.LongTensor(indices), tokens

    def fix_query_length(self, sentences):
        ls = []
        for sentence in sentences:
            ls.append(self.fix_length(sentence))
        return ls

    def fix_length(self, sentence):
        max_size = self.config.max_query_length
        if len(sentence) < max_size:
            pads = [Constants.PAD]*(max_size-len(sentence))
            return torch.LongTensor(sentence.tolist() + pads)
        else:
            return sentence[:max_size]

    def read_query_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_query_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_query_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        d = []
        for i in range(1, len(parents)+1):
            if i-1 not in trees.keys() and parents[i-1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    d.append(tree)
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx-1] = tree
                    tree.idx = idx-1
                    if parent-1 in trees.keys():
                        trees[parent-1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        if root is not None:
            root._data = d
        return root

    def load_output(self, data_dir, file_name):
        print('Reading code files...')
        trees_file = os.path.join(data_dir, '{}.out.trees.bin'.format(file_name))
        code_file = os.path.join(data_dir, '{}.out.bin'.format(file_name))
        self.code_trees = deserialize_from_file(trees_file)
        self.codes = deserialize_from_file(code_file)

        print('Constructing code representation...')
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
                        if tok_src_idx < 0 or tok_src_idx >= self.config.max_query_length:
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

    def init_data_matrices(self):
        max_example_action_num = self.config.max_example_action_num
        terminal_vocab = self.terminal_vocab

        print('Initializing data matrices...')
        self.tgt_node_seq = torch.LongTensor(self.size, max_example_action_num).zero_()
        self.tgt_par_rule_seq = torch.LongTensor(self.size, max_example_action_num).zero_()
        self.tgt_par_t_seq = torch.LongTensor(self.size, max_example_action_num).zero_()
        self.tgt_action_seq = torch.LongTensor(self.size, max_example_action_num, 3).zero_()
        self.tgt_action_seq_type = torch.LongTensor(self.size, max_example_action_num, 3).zero_()

        for eid, actions in enumerate(self.actions):
            exg_action_seq = actions[:max_example_action_num]

            assert len(exg_action_seq) > 0

            for t, action in enumerate(exg_action_seq):
                if action.act_type == APPLY_RULE:
                    rule = action.data['rule']
                    self.tgt_action_seq[eid, t, 0] = self.grammar.rule_to_id[rule]
                    self.tgt_action_seq_type[eid, t, 0] = 1
                elif action.act_type == GEN_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab.getIndex(token, Constants.UNK)
                    self.tgt_action_seq[eid, t, 1] = token_id
                    self.tgt_action_seq_type[eid, t, 1] = 1
                elif action.act_type == COPY_TOKEN:
                    src_token_idx = action.data['source_idx']
                    self.tgt_action_seq[eid, t, 2] = src_token_idx
                    self.tgt_action_seq_type[eid, t, 2] = 1
                elif action.act_type == GEN_COPY_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab.getIndex(token, Constants.UNK)
                    self.tgt_action_seq[eid, t, 1] = token_id
                    self.tgt_action_seq_type[eid, t, 1] = 1

                    src_token_idx = action.data['source_idx']
                    self.tgt_action_seq[eid, t, 2] = src_token_idx
                    self.tgt_action_seq_type[eid, t, 2] = 1
                else:
                    raise RuntimeError('wrong action type!')

                # parent information
                rule = action.data['rule']
                parent_rule = action.data['parent_rule']
                self.tgt_node_seq[eid, t] = self.grammar.get_node_type_id(rule.parent)
                if parent_rule:
                    self.tgt_par_rule_seq[eid, t] = self.grammar.rule_to_id[parent_rule]
                else:
                    assert t == 0
                    self.tgt_par_rule_seq[eid, t] = -1

                # parent hidden states
                parent_t = action.data['parent_t']
                self.tgt_par_t_seq[eid, t] = parent_t
