import os
import pandas as pd
import torch
from copy import deepcopy

from datasets.dataset import Dataset, parents_prefix
import Constants

named_variables = ['Name', 'Attack', 'Defence', 'Cost', 'Duration', 'Type', 'Player class', 'Race', 'Rarity']


# Dataset class for HS dataset
class HSDataset(Dataset):
    def __init__(self, *args):
        super(HSDataset, self).__init__(*args)

    def __getitem__(self, index):
        query_tree = deepcopy(self.query_trees[index])
        code_tree = deepcopy(self.code_trees[index])
        named_vars = deepcopy(self.named_variables[index])
        query = deepcopy(self.query[index])
        code = deepcopy(self.codes[index])
        return query_tree, code_tree, named_vars, query, code

    def load_input(self, data_dir, file_name, syntax):
        parents_file = os.path.join(data_dir, '{}.in.{}_parents'.format(file_name, parents_prefix[syntax]))
        named_vars_file = os.path.join(data_dir, '{}.in.named_vars'.format(file_name))
        tokens_file = os.path.join(data_dir, '{}.in.tokens'.format(file_name))

        print('Reading query trees...')
        self.query_trees = self.read_query_trees(parents_file)

        print('Reading query tokens...')
        self.query, self.query_tokens = self.read_query(tokens_file)
        self.query = self.fill_pads(self.query, self.query_trees)

        print('Reading named variables...')
        self.named_variables = self.read_names_vars(named_vars_file)
        self.names = torch.LongTensor(self.vocab.convertToIdx(named_variables, Constants.UNK_WORD))

    def read_names_vars(self, named_vars_file):
        named_vars = pd.read_csv(named_vars_file, index_col=0)
        named_vars = map(lambda _, r: r.tolist(), named_vars.iterrows())
        return named_vars


