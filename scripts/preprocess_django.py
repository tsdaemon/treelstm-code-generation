import shutil
import glob
from functools import reduce

from scripts.preprocess_utils import *
from lang.parse import *

import Constants

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('=' * 80)
    logging.info('Pre-processing Django dataset')
    logging.info('=' * 80)

    dj_source_dir = os.path.join(data_dir, 'card2code/en-django/')
    dj_dir = os.path.join(base_dir, 'preprocessed/django')

    if os.path.exists(dj_dir):
        shutil.rmtree(dj_dir)
    os.makedirs(dj_dir)

    # if not os.path.exists(hs_dir):
    #     os.makedirs(hs_dir)
    # else:
    #     print("Django folder found. Exiting.")
    #     exit(1)

    train_dir = os.path.join(dj_dir, 'train')
    dev_dir = os.path.join(dj_dir, 'dev')
    test_dir = os.path.join(dj_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    shutil.copy(os.path.join(dj_source_dir, 'all.anno'), os.path.join(dj_dir, 'all.anno'))
    shutil.copy(os.path.join(dj_source_dir, 'all.code'), os.path.join(dj_dir, 'all.code'))

    logging.info('Splitting dataset')
    split_input(os.path.join(dev_dir, 'dev.in'))
    split_input(os.path.join(train_dir, 'train.in'))
    split_input(os.path.join(test_dir, 'test.in'))

    logging.info('Tokenizing')
    tokenize(os.path.join(dev_dir, 'dev.in.description'))
    tokenize(os.path.join(train_dir, 'train.in.description'))
    tokenize(os.path.join(test_dir, 'test.in.description'))

    logging.info('Building vocabulary')
    vocab = build_vocab_from_token_files(glob.glob(os.path.join(hs_dir, '*/*.tokens')), min_frequency=3)
    save_vocab(os.path.join(hs_dir, 'vocab.txt'), vocab)

    logging.info('Build vocab embeddings')
    vocab = Vocab(filename=os.path.join(hs_dir, 'vocab.txt'),
                  data=[Constants.UNK_WORD, Constants.EOS_WORD, Constants.PAD_WORD])
    emb_file = os.path.join(hs_dir, 'word_embeddings.pth')
    glove_file = os.path.join(data_dir, 'glove/glove.840B.300d')
    # load glove embeddings and vocab
    glove_vocab, glove_emb = load_word_vectors(glove_file)
    emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(0.0, 0.1)
    # zero out the embeddings for padding and other special words if they are absent in vocab
    for idx, item in enumerate([Constants.UNK_WORD, Constants.EOS_WORD, Constants.PAD_WORD]):
        emb[idx].zero_()
    for word in vocab.labelToIdx.keys():
        if glove_vocab.getIndex(word):
            emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
    torch.save(emb, emb_file)

    logging.info('Parsing descriptions trees')
    parse(os.path.join(dev_dir, 'dev.in.tokens'))
    parse(os.path.join(train_dir, 'train.in.tokens'))
    parse(os.path.join(test_dir, 'test.in.tokens'))

    logging.info('Parsing output code')
    parse_trees_dev = parse_code_trees(os.path.join(dev_dir, 'dev.out'), os.path.join(dev_dir, 'dev.out.bin'))
    parse_trees_train = parse_code_trees(os.path.join(train_dir, 'train.out'), os.path.join(train_dir, 'train.out.bin'))
    parse_trees_test = parse_code_trees(os.path.join(test_dir, 'test.out'), os.path.join(test_dir, 'test.out.bin'))
    parse_trees = parse_trees_dev+parse_trees_train+parse_trees_test

    logging.info('Applying unary closures')
    do_unary_closures(parse_trees, 30)

    logging.info('Saving trees')
    write_trees(parse_trees_dev, os.path.join(dev_dir, 'dev.out.trees'))
    write_trees(parse_trees_train, os.path.join(train_dir, 'train.out.trees'))
    write_trees(parse_trees_test, os.path.join(test_dir, 'test.out.trees'))

    logging.info('Creating grammar')
    grammar = write_grammar(parse_trees, os.path.join(hs_dir, 'grammar.txt'))

    logging.info('Creating terminal vocabulary')
    write_terminal_tokens_vocab(grammar, parse_trees, os.path.join(hs_dir, 'terminal_vocab.txt'), min_freq=3)

