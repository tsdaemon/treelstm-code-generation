import shutil
import pandas as pd
import glob
import re

from scripts.preprocess_utils import *
from utils.vocab import get_glove_vocab


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
    df = pd.DataFrame(columns=['Name', 'Attack', 'Defence', 'Cost', 'Duration', 'Type', 'Player class', 'Race', 'Rarity'])
    with open(filepath, 'r') as datafile, \
         open(os.path.join(dst_dir, filepath + '.description'), 'w') as dfile:
            for line in datafile:
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


if __name__ == '__main__':
    print('=' * 80)
    print('Pre-processing Hearthstone dataset')
    print('=' * 80)

    hs_source_dir = os.path.join(data_dir, 'card2code\\third_party\\hearthstone/')
    hs_dir = os.path.join(base_dir, 'preprocessed\\hs')
    if not os.path.exists(hs_dir):
        os.makedirs(hs_dir)
    else:
        print("Hearthstone folder found. Exiting.")
        exit()

    train_dir = os.path.join(hs_dir, 'train')
    dev_dir = os.path.join(hs_dir, 'dev')
    test_dir = os.path.join(hs_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    # copy dataset
    shutil.copy(os.path.join(hs_source_dir, 'dev_hs.in'), os.path.join(dev_dir, 'dev.in'))
    shutil.copy(os.path.join(hs_source_dir, 'dev_hs.out'), os.path.join(dev_dir, 'dev.out'))
    shutil.copy(os.path.join(hs_source_dir, 'train_hs.in'), os.path.join(train_dir, 'train.in'))
    shutil.copy(os.path.join(hs_source_dir, 'train_hs.out'), os.path.join(train_dir, 'train.out'))
    shutil.copy(os.path.join(hs_source_dir, 'test_hs.in'), os.path.join(test_dir, 'test.in'))
    shutil.copy(os.path.join(hs_source_dir, 'test_hs.out'), os.path.join(test_dir, 'test.out'))

    print('Splitting dataset')
    print('=' * 80)
    split_input(os.path.join(dev_dir, 'dev.in'))
    split_input(os.path.join(train_dir, 'train.in'))
    split_input(os.path.join(test_dir, 'test.in'))

    print('Tokenizing')
    print('=' * 80)
    tokenize(os.path.join(dev_dir, 'dev.in.description'))
    tokenize(os.path.join(train_dir, 'train.in.description'))
    tokenize(os.path.join(test_dir, 'test.in.description'))

    print('Building vocabulary')
    print('=' * 80)
    vocab = build_vocab(glob.glob(os.path.join(hs_dir, '*\\*.tokens')))
    vocab_glove = get_glove_vocab().getSet()
    vocab_unk = vocab - vocab_glove
    vocab = vocab - vocab_unk
    vocab, vocab_unk = move_numbers_from_known(vocab, vocab_unk)
    save_vocab(os.path.join(hs_dir, 'vocab.txt'), vocab)
    save_vocab(os.path.join(hs_dir, 'vocab.unk.txt'), vocab_unk)

    print('Parsing descriptions for variables')
    print('=' * 80)
    parse_for_variables(os.path.join(dev_dir, 'dev.in.tokens'), vocab_unk)
    parse_for_variables(os.path.join(train_dir, 'train.in.tokens'), vocab_unk)
    parse_for_variables(os.path.join(test_dir, 'test.in.tokens'), vocab_unk)

    print('Parsing descriptions trees')
    print('=' * 80)
    parse(os.path.join(dev_dir, 'dev.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))
    parse(os.path.join(train_dir, 'train.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))
    parse(os.path.join(test_dir, 'test.in.tokens'), os.path.join(hs_dir, 'vocab.unk.txt'))

