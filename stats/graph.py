import os

from natural_lang.tree import *
from config import parser


def read_one_line_from_file(file):
    with open(file, 'r') as f:
        return f.readline()

if __name__ == '__main__':
    args = parser.parse_args()
    hs_tree_path = os.path.join(args.data_dir, 'test/test.in.ccg_parents')
    hs_category_path = os.path.join(args.data_dir, 'test/test.in.ccg_categories')
    hs_tokens_path = os.path.join(args.data_dir, 'test/test.in.tokens')

    hs_tree_line = read_one_line_from_file(hs_tree_path)
    hs_tokens = read_one_line_from_file(hs_tokens_path).split()
    hs_categories = read_one_line_from_file(hs_category_path).split()
    hs_tree = read_tree(hs_tree_line)
    hs_tree.plot(hs_tokens, hs_categories)

