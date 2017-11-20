import os
import numpy as np
import logging
import sys
import torch
import random

from utils.general import init_logging
from model.x2x import Tree2TreeModel
from config import parser


if __name__ == '__main__':
    args = parser.parse_args()

    # arguments validation
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.benchmark = True

    # prepare dirs
    data_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # start logging
    init_logging(os.path.join(args.output_dir, 'parser.log'), logging.INFO)
    logging.info('command line: %s', ' '.join(sys.argv))
    logging.info('current config: %s', args)
    logging.info('loading dataset [%s]', args.data)

    # load data
    load_dataset = None
    if args.dataset == 'hs':
        from datasets.hs import load_dataset
    else:
        raise Exception('Dataset {} is not prepared yet'.format(args.dataset))
    train_data, dev_data, test_data = load_dataset(args)

    # configure more variables
    args.source_vocab_size = train_data.annot_vocab.size
    args.target_vocab_size = train_data.terminal_vocab.size
    args.rule_num = len(train_data.grammar.rules)
    args.node_num = len(train_data.grammar.node_type_to_id)

    # load model
    emb_file = os.path.join(data_dir, 'word_embeddings.pth')
    emb = torch.load(emb_file)
    if args.cuda:
        emb = emb.cuda()
    model = Tree2TreeModel(args, emb)
    if args.cuda:
        model = model.cuda()

    # create learner
    learner = Learner(model, train_data, dev_data)
    learner.train()