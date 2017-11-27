import shutil
import glob
import astor
from tqdm import tqdm
from functools import reduce

from scripts.preprocess_utils import *
from natural_lang.vocab import get_glove_vocab
from lang.parse import *
from utils.io import serialize_to_file
from lang.unaryclosure import apply_unary_closures, get_top_unary_closures
import Constants

position_symbols = ["NAME_END",
                    "ATK_END",
                    "DEF_END",
                    "COST_END",
                    "DUR_END",
                    "TYPE_END",
                    "PLAYER_CLS_END",
                    "RACE_END",
                    "RARITY_END"]

names = ['Name', 'attack', 'defence', 'cost', 'duration', 'type', 'player class', 'race', 'rarity']


def extract_from_hs_line(line, end_symbol, start_pos=None):
    if start_pos is None:
        start_pos = 0

    end_pos = line.find(" " + end_symbol)
    result = line[start_pos:end_pos]
    new_pos = end_pos + len(end_symbol) + 2
    return result, new_pos


def tranform_description(vars, desc):
    vars_desc = map(lambda t: '{}: {}'.format(t[0], t[1]), zip(names, vars))
    vars_line = reduce(lambda v1, v2: '{}, {}'.format(v1, v2), vars_desc) + "."

    if desc == "NIL\n":
        desc = vars_line + "\n"
    else:
        desc = vars_line + " " + desc
    return re.sub(r"<[^>]*>", "", desc)


def split_input(filepath):
    print('Splitting input ' + filepath)
    dst_dir = os.path.dirname(filepath)
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

                description = tranform_description(vars, line[position:])
                dfile.write(description)


def parse_code_trees(code_file, code_out_file):
    print('Parsing code trees from file {}'.format(code_file))
    parse_trees = []
    codes = []
    rule_num = 0.
    example_num = 0
    for line in tqdm(open(code_file).readlines()):
        lb = 'В§' if system == 'w' else '§'
        code = line.replace(lb, '\n').replace('    ', '\t')
        code = canonicalize_code(code)
        codes.append(code)

        p_tree = parse_raw(code)
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

    serialize_to_file(codes, code_out_file)
    return parse_trees


def write_grammar(parse_trees, out_file):
    grammar = get_grammar(parse_trees)

    serialize_to_file(grammar, out_file + '.bin')
    with open(out_file, 'w') as f:
        for rule in tqdm(grammar):
            str = rule.__repr__()
            f.write(str + '\n')

    return grammar


def write_terminal_tokens_vocab(grammar, parse_trees, out_file):
    terminal_token_seq = []

    for parse_tree in tqdm(parse_trees):
        for node in parse_tree.get_leaves():
            if grammar.is_value_node(node):
                terminal_val = node.value
                terminal_str = str(terminal_val)

                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    terminal_token_seq.append(terminal_token)

    terminal_vocab = build_vocab_from_items(terminal_token_seq, False)
    save_vocab(out_file, terminal_vocab)


def do_unary_closures(parse_trees):
    print('Applying unary closures to parse trees...')
    unary_closures = get_top_unary_closures(parse_trees, k=20)
    for parse_tree in tqdm(parse_trees):
        apply_unary_closures(parse_tree, unary_closures)


def write_trees(parse_trees, out_file):
    # save data
    with open(out_file, 'w') as f:
        for tree in tqdm(parse_trees):
            f.write(tree.__repr__() + '\n')
    serialize_to_file(parse_trees, out_file + '.bin')


if __name__ == '__main__':
    print('=' * 80)
    print('Pre-processing HearthStone dataset')
    print('=' * 80)

    hs_source_dir = os.path.join(data_dir, 'card2code/third_party/hearthstone/')
    hs_dir = os.path.join(base_dir, 'preprocessed/hs')

    # if os.path.exists(hs_dir):
    #     shutil.rmtree(hs_dir)
    # os.makedirs(hs_dir)

    # if not os.path.exists(hs_dir):
    #     os.makedirs(hs_dir)
    # else:
    #     print("Hearthstone folder found. Exiting.")
    #     exit(1)

    train_dir = os.path.join(hs_dir, 'train')
    dev_dir = os.path.join(hs_dir, 'dev')
    test_dir = os.path.join(hs_dir, 'test')
    # make_dirs([train_dir, dev_dir, test_dir])

    # copy dataset
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
    # vocab = build_vocab_from_token_files(glob.glob(os.path.join(hs_dir, '*/*.tokens')))
    # vocab_glove = get_glove_vocab().getSet()
    # vocab_unk = vocab - vocab_glove
    # vocab = vocab - vocab_unk
    # vocab, vocab_unk = move_numbers_from_known(vocab, vocab_unk)
    # save_vocab(os.path.join(hs_dir, 'vocab.txt'), vocab)
    # save_vocab(os.path.join(hs_dir, 'vocab.unk.txt'), vocab_unk)
    #
    # print('Build vocab embeddings')
    # vocab = Vocab(filename=os.path.join(hs_dir, 'vocab.txt'), data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    # emb_file = os.path.join(hs_dir, 'word_embeddings.pth')
    # glove_file = os.path.join(data_dir, 'glove/glove.840B.300d')
    # # load glove embeddings and vocab
    # glove_vocab, glove_emb = load_word_vectors(glove_file)
    # emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(-0.05, 0.05)
    # # zero out the embeddings for padding and other special words if they are absent in vocab
    # for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
    #     emb[idx].zero_()
    # for word in vocab.labelToIdx.keys():
    #     if glove_vocab.getIndex(word):
    #         emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
    # torch.save(emb, emb_file)
    #
    # # print('Parsing descriptions for variables')
    # # parse_for_variables(os.path.join(dev_dir, 'dev.in.tokens'), vocab_unk)
    # # parse_for_variables(os.path.join(train_dir, 'train.in.tokens'), vocab_unk)
    # # parse_for_variables(os.path.join(test_dir, 'test.in.tokens'), vocab_unk)
    #
    # print('Parsing descriptions trees')
    # parse(os.path.join(dev_dir, 'dev.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))
    # parse(os.path.join(train_dir, 'train.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))
    # parse(os.path.join(test_dir, 'test.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))

    print('Parsing output code')
    parse_trees_dev = parse_code_trees(os.path.join(dev_dir, 'dev.out'), os.path.join(dev_dir, 'dev.out.bin'))
    parse_trees_train = parse_code_trees(os.path.join(train_dir, 'train.out'), os.path.join(train_dir, 'train.out.bin'))
    parse_trees_test = parse_code_trees(os.path.join(test_dir, 'test.out'), os.path.join(test_dir, 'test.out.bin'))
    parse_trees = parse_trees_dev+parse_trees_train+parse_trees_test

    print('Applying unary closures')
    do_unary_closures(parse_trees)

    print('Saving trees')
    write_trees(parse_trees_dev, os.path.join(dev_dir, 'dev.out.trees'))
    write_trees(parse_trees_train, os.path.join(train_dir, 'train.out.trees'))
    write_trees(parse_trees_test, os.path.join(test_dir, 'test.out.trees'))

    print('Creating grammar')
    grammar = write_grammar(parse_trees, os.path.join(hs_dir, 'grammar.txt'))

    print('Creating terminal vocabulary')
    write_terminal_tokens_vocab(grammar, parse_trees, os.path.join(hs_dir, 'terminal_vocab.txt'))

