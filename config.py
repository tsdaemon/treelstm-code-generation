import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='./preprocessed')
parser.add_argument('-random_seed', default=181783, type=int)
parser.add_argument('-output_dir', default='./results')

# experiment's main configuration
parser.add_argument('-dataset', default='django', choices=['django', 'hs', 'bs'])
parser.add_argument('-model', default='', type=str)
parser.add_argument('-mode', default='train', choices=['train', 'validate'])

# neural model's parameters
parser.add_argument('-word_embed_dim', default=300, type=int)
parser.add_argument('-rule_embed_dim', default=256, type=int)
parser.add_argument('-node_embed_dim', default=256, type=int)
parser.add_argument('-encoder_hidden_dim', default=256, type=int)
parser.add_argument('-decoder_hidden_dim', default=256, type=int)
parser.add_argument('-attention_hidden_dim', default=50, type=int)
parser.add_argument('-ptrnet_hidden_dim', default=50, type=int)

# encoder
parser.add_argument('-encoder', default='recursive-lstm', choices=['recursive-lstm', 'bi-lstm', 'bi-lstm-dropout'])
parser.add_argument('-syntax', default='ccg', choices=['ccg', 'pcfg', 'dependency'])
parser.add_argument('-pretrained_embeds', dest='pretrained_embeds', action='store_true')
parser.add_argument('-no_pretrained_embeds', dest='pretrained_embeds', action='store_false')
parser.set_defaults(pretrained_embeds=True)
parser.add_argument('-freeze_embeds', dest='freeze_embeds', action='store_true')
parser.add_argument('-no_freeze_embeds', dest='freeze_embeds', action='store_false')
parser.set_defaults(freeze_embeds=False)
parser.add_argument('-encoder_dropout', default=0.2, type=float)

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

parser.add_argument('-decoder_dropout', default=0.2, type=float)

# x2x
parser.add_argument('-thought_vector', dest='thought_vector', action='store_true')
parser.add_argument('-no_thought_vector', dest='thought_vector', action='store_false')
parser.set_defaults(thought_vector=True)

parser.add_argument('-unary_closures', dest='unary_closures', action='store_true')
parser.add_argument('-no_unary_closures', dest='unary_closures', action='store_false')
parser.set_defaults(unary_closures=True)

# training
parser.add_argument('-clip_grad', default=0., type=float)
parser.add_argument('-train_patience', default=10, type=int)
parser.add_argument('-max_epoch', default=100, type=int)
parser.add_argument('-batch_size', default=10, type=int)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-cuda', dest='cuda', action='store_true')
parser.add_argument('-no_cuda', dest='cuda', action='store_false')
parser.add_argument('-valid_metric', default='bleu')

# decoding
parser.add_argument('-beam_size', default=10, type=int)
# parser.add_argument('-max_query_length', default=200, type=int)
# parser.add_argument('-max_example_action_num', default=200, type=int)
# parser.add_argument('-decode_max_time_step', default=200, type=int)
parser.add_argument('-max_query_length', default=70, type=int)
parser.add_argument('-max_example_action_num', default=100, type=int)
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