import os
import torch
import logging

from datasets.dataset import Dataset, parents_prefix
import Constants
from utils.io import deserialize_from_file
from natural_lang.vocab import Vocab
from config import parser


def load_dataset(config, force_regenerate=False):
    data_dir = config.data_dir
    hs_dir = os.path.join(data_dir, 'hs')
    logging.info('='*80)
    logging.info('Loading dataset from folder ' + hs_dir)
    logging.info('='*80)
    train, test, dev = None, None, None

    train_dir = os.path.join(hs_dir, 'train')
    train_file = os.path.join(train_dir, 'train.pth')
    if not force_regenerate and os.path.isfile(train_file):
        logging.info('Train dataset found, loading...')
        train = torch.load(train_file)

    test_dir = os.path.join(hs_dir, 'test')
    test_file = os.path.join(test_dir, 'test.pth')
    if not force_regenerate and os.path.isfile(test_file):
        logging.info('Test dataset found, loading...')
        test = torch.load(test_file)

    dev_dir = os.path.join(hs_dir, 'dev')
    dev_file = os.path.join(dev_dir, 'dev.pth')
    if not force_regenerate and os.path.isfile(dev_file):
        logging.info('Dev dataset found, loading...')
        dev = torch.load(dev_file)

    if train is None or test is None or dev is None:
        grammar = deserialize_from_file(os.path.join(hs_dir, 'grammar.txt.bin'))
        terminal_vocab = Vocab(os.path.join(hs_dir, 'terminal_vocab.txt'), data=[Constants.UNK_WORD, Constants.EOS_WORD, Constants.PAD_WORD])
        vocab = Vocab(os.path.join(hs_dir, 'vocab.txt'), data=[Constants.UNK_WORD, Constants.EOS_WORD, Constants.PAD_WORD])

        if test is None:
            print('Test dataset not found, generating...')
            test = Dataset(test_dir, 'test', grammar, vocab, terminal_vocab, config)
            torch.save(test, test_file)

        if dev is None:
            print('Dev dataset not found, generating...')
            dev = Dataset(dev_dir, 'dev', grammar, vocab, terminal_vocab, config)
            torch.save(dev, dev_file)

        if train is None:
            print('Train dataset not found, generating...')
            train = Dataset(train_dir, 'train', grammar, vocab, terminal_vocab, config)
            torch.save(train, train_file)

    train.prepare_torch()
    dev.prepare_torch()
    test.prepare_torch()
    return train, dev, test


if __name__ == '__main__':
    config = parser.parse_args()
    load_dataset(config, force_regenerate=True)
