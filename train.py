import argparse
import os
import numpy as np
import logging
import sys

from utils.general import init_logging
from utils.io import deserialize_from_file, serialize_to_file
from learner import Learner
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('-data')
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-output_dir', default='.outputs')

parser.add_argument('-data_type', default='django', choices=['bs', 'ifttt', 'hs'])

parser.add_argument('-word_embed_dim', default=128, type=int)
parser.add_argument('-rule_embed_dim', default=256, type=int)
parser.add_argument('-node_embed_dim', default=256, type=int)
parser.add_argument('-encoder_hidden_dim', default=256, type=int)
parser.add_argument('-decoder_hidden_dim', default=256, type=int)
parser.add_argument('-attention_hidden_dim', default=50, type=int)
parser.add_argument('-ptrnet_hidden_dim', default=50, type=int)
parser.add_argument('-dropout', default=0.2, type=float)

# decoder
parser.add_argument('-parent_hidden_state_feed', dest='parent_hidden_state_feed', action='store_true')
parser.add_argument('-no_parent_hidden_state_feed', dest='parent_hidden_state_feed', action='store_false')
parser.set_defaults(parent_hidden_state_feed=True)

parser.add_argument('-parent_action_feed', dest='parent_action_feed', action='store_true')
parser.add_argument('-no_parent_action_feed', dest='parent_action_feed', action='store_false')
parser.set_defaults(parent_action_feed=True)

parser.add_argument('-frontier_node_type_feed', dest='frontier_node_type_feed', action='store_true')
parser.add_argument('-no_frontier_node_type_feed', dest='frontier_node_type_feed', action='store_false')
parser.set_defaults(frontier_node_type_feed=True)

parser.add_argument('-tree_attention', dest='tree_attention', action='store_true')
parser.add_argument('-no_tree_attention', dest='tree_attention', action='store_false')
parser.set_defaults(tree_attention=False)

parser.add_argument('-enable_copy', dest='enable_copy', action='store_true')
parser.add_argument('-no_copy', dest='enable_copy', action='store_false')
parser.set_defaults(enable_copy=True)

# training
parser.add_argument('-optimizer', default='adam')
parser.add_argument('-clip_grad', default=0., type=float)
parser.add_argument('-train_patience', default=10, type=int)
parser.add_argument('-max_epoch', default=50, type=int)
parser.add_argument('-batch_size', default=10, type=int)
parser.add_argument('-valid_per_batch', default=4000, type=int)
parser.add_argument('-save_per_batch', default=4000, type=int)
parser.add_argument('-valid_metric', default='bleu')


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.random.seed(args.random_seed)
    init_logging(os.path.join(args.output_dir, 'parser.log'), logging.INFO)
    logging.info('command line: %s', ' '.join(sys.argv))

    logging.info('loading dataset [%s]', args.data)
    train_data, dev_data, test_data = deserialize_from_file(args.data)

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