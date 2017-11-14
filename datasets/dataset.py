from copy import deepcopy
import tqdm
import torch.utils.data as data
import torch

import Constants
from containers.tree import Tree


parents_prefix = {
    'ccg': 'ccg',
    'pcfg': 'constituency',
    'dependency': 'dependency'
}


class Dataset(data.Dataset):
    def __init__(self, data_dir, file_name, vocab, terminal_vocab, syntax):
        super(Dataset, self).__init__()
        self.vocab = vocab
        self.terminal_vocab = terminal_vocab

        self.load_input(data_dir, file_name, syntax)
        self.size = self.load_output(data_dir, file_name)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        enc_tree = deepcopy(self.enc_trees[index])
        dec_tree = deepcopy(self.dec_trees[index])
        input = deepcopy(self.inputs[index])
        code = deepcopy(self.codes[index])
        return enc_tree, dec_tree, input, code

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def fill_pads(self, sentences, trees):
        ls = []
        for sentence, tree in zip(sentences, trees):
            ls.append(self.fill_pad(sentence, tree))
        return ls

    def fill_pad(self, sentence, tree):
        tree_size = tree.size()
        if len(sentence) < tree_size:
            pads = [Constants.PAD]*(tree_size-len(sentence))
            return torch.LongTensor(sentence.tolist() + pads)
        else:
            return sentence

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        d = [root]
        for i in range(1, len(parents)+1):
            if i-1 not in trees.keys() and parents[i-1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    data.append(tree)
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
        root._data = d
        return root
