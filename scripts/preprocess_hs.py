import shutil
import pandas as pd
import glob
import ast
import astor
from tqdm import tqdm

from datasets.hs import named_variables
from scripts.preprocess_utils import *
from containers.vocab import get_glove_vocab
from lang.parse import parse_code, parse_tree_to_python_ast, get_grammar


position_symbols = ["NAME_END",
                    "ATK_END",
                    "DEF_END",
                    "COST_END",
                    "DUR_END",
                    "TYPE_END",
                    "PLAYER_CLS_END",
                    "RACE_END",
                    "RARITY_END"]


def extract_from_hs_line(line, end_symbol, start_pos=None):
    if start_pos is None:
        start_pos = 0

    end_pos = line.find(" " + end_symbol)
    result = line[start_pos:end_pos]
    new_pos = end_pos + len(end_symbol) + 2
    return result, new_pos


def tranform_description(desc):
   if desc == "NIL\n":
        return "\n"
   return re.sub(r"<[^>]*>", "", desc)


def split_input(filepath):
    print('\nSplitting input ' + filepath)
    dst_dir = os.path.dirname(filepath)
    df = pd.DataFrame(columns=named_variables)
    with open(filepath, 'r') as datafile, \
         open(os.path.join(dst_dir, filepath + '.description'), 'w') as dfile:
            for line in tqdm(datafile.readlines()):
                vars = []
                position = 0
                for pos_sym in position_symbols:
                    var, position = extract_from_hs_line(line, pos_sym, position)
                    if var == "NIL":
                        var = "None"
                    vars.append(var)
                df.loc[len(df)] = vars

                description = tranform_description(line[position:])

                dfile.write(description)
    df.to_csv(os.path.join(dst_dir, filepath + '.named_vars'))


def parse_code_trees(code_file, output_file):
    parse_trees = []
    rule_num = 0.
    example_num = 0
    for line in tqdm(open(code_file).readlines()):
        code = line.replace('В§', '\n').replace('    ', '\t')
        p_tree = parse_code(code)
        # sanity check
        pred_ast = parse_tree_to_python_ast(p_tree)
        pred_code = astor.to_source(pred_ast)
        ref_ast = ast.parse(code)
        ref_code = astor.to_source(ref_ast)

        if pred_code != ref_code:
            raise RuntimeError('code mismatch!')

        rules, _ = p_tree.get_productions(include_value_node=False)
        rule_num += len(rules)
        example_num += 1

        parse_trees.append(p_tree)

    with open(output_file, 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')

    print('Avg. nums of rules: %f' % (rule_num / example_num))
    return parse_trees


def write_grammar(parse_trees, out_file):
    grammar = get_grammar(parse_trees)

    with open(out_file, 'w') as f:
        for rule in tqdm(grammar):
            str = rule.__repr__()
            f.write(str + '\n')
    return grammar


if __name__ == '__main__':
    print('=' * 80)
    print('Pre-processing HearthStone dataset')
    print('=' * 80)

    hs_source_dir = os.path.join(data_dir, 'card2code\\third_party\\hearthstone/')
    hs_dir = os.path.join(base_dir, 'preprocessed\\hs')
    # if not os.path.exists(hs_dir):
    #     os.makedirs(hs_dir)
    # else:
    #     print("Hearthstone folder found. Exiting.")
    #     exit()
    #
    train_dir = os.path.join(hs_dir, 'train')
    dev_dir = os.path.join(hs_dir, 'dev')
    test_dir = os.path.join(hs_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])
    #
    # # copy dataset
    # shutil.copy(os.path.join(hs_source_dir, 'dev_hs.in'), os.path.join(dev_dir, 'dev.in'))
    # shutil.copy(os.path.join(hs_source_dir, 'dev_hs.out'), os.path.join(dev_dir, 'dev.out'))
    # shutil.copy(os.path.join(hs_source_dir, 'train_hs.in'), os.path.join(train_dir, 'train.in'))
    # shutil.copy(os.path.join(hs_source_dir, 'train_hs.out'), os.path.join(train_dir, 'train.out'))
    # shutil.copy(os.path.join(hs_source_dir, 'test_hs.in'), os.path.join(test_dir, 'test.in'))
    # shutil.copy(os.path.join(hs_source_dir, 'test_hs.out'), os.path.join(test_dir, 'test.out'))
    #
    # print('Splitting dataset')
    # split_input(os.path.join(dev_dir, 'dev.in'))
    # split_input(os.path.join(train_dir, 'train.in'))
    # split_input(os.path.join(test_dir, 'test.in'))
    #
    # print('Tokenizing')
    # tokenize(os.path.join(dev_dir, 'dev.in.description'))
    # tokenize(os.path.join(train_dir, 'train.in.description'))
    # tokenize(os.path.join(test_dir, 'test.in.description'))
    #
    # print('Building vocabulary')
    # vocab = build_vocab(glob.glob(os.path.join(hs_dir, '*/*.tokens')))
    # vocab_glove = get_glove_vocab().getSet()
    # vocab_unk = vocab - vocab_glove
    # vocab = vocab - vocab_unk
    # vocab, vocab_unk = move_numbers_from_known(vocab, vocab_unk)
    # save_vocab(os.path.join(hs_dir, 'vocab.txt'), vocab)
    # save_vocab(os.path.join(hs_dir, 'vocab.unk.txt'), vocab_unk)
    #
    # print('Parsing descriptions for variables')
    # parse_for_variables(os.path.join(dev_dir, 'dev.in.tokens'), vocab_unk)
    # parse_for_variables(os.path.join(train_dir, 'train.in.tokens'), vocab_unk)
    # parse_for_variables(os.path.join(test_dir, 'test.in.tokens'), vocab_unk)
    #
    # print('Parsing descriptions trees')
    # parse(os.path.join(dev_dir, 'dev.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))
    # parse(os.path.join(train_dir, 'train.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))
    # parse(os.path.join(test_dir, 'test.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))

    print('Parsing output code')
    parse_trees_dev = parse_code_trees(os.path.join(dev_dir, 'dev.out'), os.path.join(dev_dir, 'dev.out.trees'))
    parse_trees_train = parse_code_trees(os.path.join(train_dir, 'train.out'), os.path.join(train_dir, 'train.out.trees'))
    parse_trees_test = parse_code_trees(os.path.join(test_dir, 'test.out'), os.path.join(test_dir, 'test.out.trees'))

    write_grammar(parse_trees_dev+parse_trees_train+parse_trees_test, os.path.join(hs_dir, 'grammar.txt'))

