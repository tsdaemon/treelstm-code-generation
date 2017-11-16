import os
import numpy as np
import logging
import sys

from utils.general import init_logging
from model.x2x import Tree2TreeModel
from config import parser


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.random.seed(args.random_seed)
    init_logging(os.path.join(args.output_dir, 'parser.log'), logging.INFO)
    logging.info('command line: %s', ' '.join(sys.argv))

    logging.info('loading dataset [%s]', args.data)

    load_dataset = None

    if args.dataset == 'hs':
        from datasets.hs import load_dataset
    else:
        raise Exception('Dataset {} is not prepared yet'.format(args.dataset))

    train_data, dev_data, test_data = load_dataset(args)

    if not args.source_vocab_size:
        args.source_vocab_size = train_data.vocab.size
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

    model = Tree2TreeModel(args)

    learner = Learner(model, train_data, dev_data)
    learner.train()