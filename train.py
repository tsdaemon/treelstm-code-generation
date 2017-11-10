import argparse
import os
import numpy as np
import logging
import sys

from utils.general import init_logging
from utils.io import deserialize_from_file, serialize_to_file
from learner import Learner
from model import Model
from config import train_parser


if __name__ == '__main__':
    args = train_parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.random.seed(args.random_seed)
    init_logging(os.path.join(args.output_dir, 'parser.log'), logging.INFO)
    logging.info('command line: %s', ' '.join(sys.argv))

    logging.info('loading dataset [%s]', args.data)
    train_data, dev_data, test_data = deserialize_from_file(args.data)

    if not args.source_vocab_size:
        args.source_vocab_size = train_data.annot_vocab.size
    if not args.target_vocab_size:
        args.target_vocab_size = train_data.terminal_vocab.size
    if not args.rule_num:
        args.rule_num = len(train_data.grammar.rules)
    if not args.node_num:
        args.node_num = len(train_data.grammar.node_type_to_id)

    args.source_vocab_size = train_data.annot_vocab.size
    args.target_vocab_size = train_data.terminal_vocab.size
    args.rule_num = len(train_data.grammar.rules)
    args.node_num = len(train_data.grammar.node_type_to_id)

    logging.info('current config: %s', args)
    # config_module = sys.modules['config']
    # for name, value in vars(args).iteritems():
    #     setattr(config_module, name, value)

    model = Model(args)
    model.build()

    # train_data = train_data.get_dataset_by_ids(range(2000), 'train_sample')
    # dev_data = dev_data.get_dataset_by_ids(range(10), 'dev_sample')
    learner = Learner(model, train_data, dev_data)
    learner.train()