import os
from functools import reduce

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


def draw_trees(data_dir, line):
    hs_tree_path = os.path.join(data_dir, 'dev/dev.in.constituency_parents')
    hs_category_path = os.path.join(data_dir, 'dev/dev.in.constituency_categories')
    hs_tokens_path = os.path.join(data_dir, 'dev/dev.in.tokens')
    out_path = os.path.join(data_dir, 'dev/pcfg_tree_example.png')

    draw_tree(hs_tree_path, hs_category_path, hs_tokens_path, line, out_path)

    hs_tree_path = os.path.join(data_dir, 'dev/dev.in.dependency_parents')
    hs_category_path = os.path.join(data_dir, 'dev/dev.in.dependency_rels')
    out_path = os.path.join(data_dir, 'dev/dependency_tree_example.png')

    draw_tree(hs_tree_path, hs_category_path, hs_tokens_path, line, out_path)

    hs_tree_path = os.path.join(data_dir, 'dev/dev.in.ccg_parents')
    hs_category_path = os.path.join(data_dir, 'dev/dev.in.ccg_categories')
    out_path = os.path.join(data_dir, 'dev/ccg_tree_example.png')

    draw_tree(hs_tree_path, hs_category_path, hs_tokens_path, line, out_path)


def avg_nodes(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        count = len(lines)
        count_nodes = sum([len(line.split()) for line in lines])
    return count, count_nodes


def avg_nodes_dataset(data_dir):
    splits = ['train', 'test', 'dev']

    nodes = [1.e-7, 0.0]
    for split in splits:
        pcfg_file = os.path.join(data_dir, '{}/{}.in.constituency_parents'.format(split, split))
        if os.path.exists(pcfg_file):
            count, count_nodes = avg_nodes(pcfg_file)
            nodes[0] += count
            nodes[1] += count_nodes
    pcfg_avg = nodes[1]/nodes[0]

    nodes = [1.e-7, 0.0]
    for split in splits:
        dependency_file = os.path.join(data_dir, '{}/{}.in.dependency_parents'.format(split, split))
        if os.path.exists(dependency_file):
            count, count_nodes = avg_nodes(dependency_file)
            nodes[0] += count
            nodes[1] += count_nodes
    dependency_avg = nodes[1] / nodes[0]

    nodes = [1.e-7, 0.0]
    for split in splits:
        ccg_file = os.path.join(data_dir, '{}/{}.in.ccg_parents'.format(split, split))
        if os.path.exists(ccg_file):
            count, count_nodes = avg_nodes(ccg_file)
            nodes[0] += count
            nodes[1] += count_nodes
    ccg_avg = nodes[1] / nodes[0]
    return dependency_avg, pcfg_avg, ccg_avg


if __name__ == '__main__':
    args = parser.parse_args()
    # draw_trees(args.data_dir)

    dependency_avg, pcfg_avg, ccg_avg = avg_nodes_dataset(args.data_dir)
    print("Average nodes in dependency trees: {}\n"
          "Average nodes in constituency trees: {}\n"
          "Average nodes in CCG trees: {}".format(dependency_avg, pcfg_avg, ccg_avg))




