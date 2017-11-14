import os
import pandas as pd
import torch
from copy import deepcopy

from datasets.dataset import Dataset, parents_prefix
import Constants

named_variables = ['Name', 'Attack', 'Defence', 'Cost', 'Duration', 'Type', 'Player class', 'Race', 'Rarity']


# Dataset class for SICK dataset
class HSDataset(Dataset):
    def __init__(self, *args):
        super(HSDataset, self).__init__(*args)

    def __getitem__(self, index):
        enc_tree = deepcopy(self.enc_trees[index])
        dec_tree = deepcopy(self.dec_trees[index])
        named_vars = deepcopy(self.loc[index].tolist())
        input = deepcopy(self.inputs[index])
        code = deepcopy(self.codes[index])
        return enc_tree, dec_tree, named_vars, input, code

    def load_input(self, data_dir, file_name, syntax):
        parents_file = os.path.join(data_dir, '{}.in.{}_parents'.format(file_name, parents_prefix[syntax]))
        named_vars_file = os.path.join(data_dir, '{}.in.named_vars'.format(file_name))
        tokens_file = os.path.join(data_dir, '{}.in.tokens')

        self.enc_trees = self.read_trees(parents_file)

        self.inputs = self.read_sentences(tokens_file)
        self.inputs = self.fill_pads(self.inputs, self.trees)

        self.named_variables = self.read_names_vars(named_vars_file)
        self.names = torch.LongTensor(self.vocab.convertToIdx(named_variables, Constants.UNK_WORD))

    def read_names_vars(self, named_vars_file):
        named_vars = pd.read_csv(named_vars_file, index_col=0)
        return named_vars

    def load_output(self, data_dir, file_name):
        pass
