import os

from natural_lang.tree import *
from config import parser


def read_line_from_file(file, line):
    with open(file, 'r') as f:
        _ = [f.readline() for _ in range(line)]
        return f.readline()


def draw_tree(parents_path, cat_path, token_path, line, out_path):
    hs_tree_line = read_line_from_file(parents_path, line)
    hs_tokens = read_line_from_file(token_path, line).split()
    hs_categories = read_line_from_file(cat_path, line).split()

    hs_labels = []
    for i in range(len(hs_categories)):
        label = hs_categories[i]
        if len(hs_tokens) > i and label != hs_tokens[i]:
            label += ' - ' + hs_tokens[i]
        hs_labels.append(label)

    hs_tree = read_tree(hs_tree_line, hs_labels)
    hs_tree.savefig(out_path)


if __name__ == '__main__':
    args = parser.parse_args()
    hs_tree_path = os.path.join(args.data_dir, 'dev/dev.in.constituency_parents')
    hs_category_path = os.path.join(args.data_dir, 'dev/dev.in.constituency_categories')
    hs_tokens_path = os.path.join(args.data_dir, 'dev/dev.in.tokens')
    out_path = os.path.join(args.data_dir, 'dev/pcfg_tree_example.png')
    line = 2

    draw_tree(hs_tree_path, hs_category_path, hs_tokens_path, line, out_path)

    hs_tree_path = os.path.join(args.data_dir, 'dev/dev.in.dependency_parents')
    hs_category_path = os.path.join(args.data_dir, 'dev/dev.in.dependency_rels')
    out_path = os.path.join(args.data_dir, 'dev/dependency_tree_example.png')

    draw_tree(hs_tree_path, hs_category_path, hs_tokens_path, line, out_path)

    hs_tree_path = os.path.join(args.data_dir, 'dev/dev.in.ccg_parents')
    hs_category_path = os.path.join(args.data_dir, 'dev/dev.in.ccg_categories')
    out_path = os.path.join(args.data_dir, 'dev/ccg_tree_example.png')

    draw_tree(hs_tree_path, hs_category_path, hs_tokens_path, line, out_path)



