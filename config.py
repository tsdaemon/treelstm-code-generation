import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='./preprocessed')
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-output_dir', default='checkpoints')

# model's main configuration
parser.add_argument('-dataset', default='django', choices=['django', 'hs', 'bs'])

# neural model's parameters
parser.add_argument('-source_vocab_size', default=0, type=int)
parser.add_argument('-target_vocab_size', default=0, type=int)
parser.add_argument('-rule_num', default=0, type=int)
parser.add_argument('-node_num', default=0, type=int)

parser.add_argument('-word_embed_dim', default=300, type=int)
parser.add_argument('-rule_embed_dim', default=256, type=int)
parser.add_argument('-node_embed_dim', default=256, type=int)
parser.add_argument('-encoder_hidden_dim', default=256, type=int)
parser.add_argument('-decoder_hidden_dim', default=256, type=int)
parser.add_argument('-attention_hidden_dim', default=50, type=int)
parser.add_argument('-ptrnet_hidden_dim', default=50, type=int)
parser.add_argument('-dropout', default=0.2, type=float)

# encoder
parser.add_argument('-syntax', default='ccg', choices=['ccg', 'pcfg', 'dependency'])

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

# decoding
parser.add_argument('-beam_size', default=15, type=int)
parser.add_argument('-max_query_length', default=70, type=int)
parser.add_argument('-max_example_action_num', default=200, type=int)
parser.add_argument('-decode_max_time_step', default=100, type=int)
parser.add_argument('-head_nt_constraint', dest='head_nt_constraint', action='store_true')
parser.add_argument('-no_head_nt_constraint', dest='head_nt_constraint', action='store_false')
parser.set_defaults(head_nt_constraint=True)

sub_parsers = parser.add_subparsers(dest='operation', help='operation to take')
train_parser = sub_parsers.add_parser('train')
decode_parser = sub_parsers.add_parser('decode')
interactive_parser = sub_parsers.add_parser('interactive')
evaluate_parser = sub_parsers.add_parser('evaluate')

# decoding operation
decode_parser.add_argument('-saveto', default='decode_results.bin')
decode_parser.add_argument('-type', default='test_data')

# evaluation operation
evaluate_parser.add_argument('-mode', default='self')
evaluate_parser.add_argument('-input', default='decode_results.bin')
evaluate_parser.add_argument('-type', default='test_data')
evaluate_parser.add_argument('-seq2tree_sample_file', default='model.sample')
evaluate_parser.add_argument('-seq2tree_id_file', default='test.id.txt')
evaluate_parser.add_argument('-seq2tree_rareword_map', default=None)
evaluate_parser.add_argument('-seq2seq_decode_file')
evaluate_parser.add_argument('-seq2seq_ref_file')
evaluate_parser.add_argument('-is_nbest', default=False, action='store_true')

# interactive operation
interactive_parser.add_argument('-mode', default='dataset')