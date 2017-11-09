import sys
from torch import nn

sys.setrecursionlimit(50000)

class Model(nn.Module):
    def __init__(self, config):
        # self.node_embedding = Embedding(config.node_num, config.node_embed_dim, name='node_embed')

        self.query_embedding = nn.Embedding(config.source_vocab_size, config.word_embed_dim, name='query_embed')

        if config.encoder == 'bilstm':
            self.query_encoder_lstm = BiLSTM(config.word_embed_dim, config.encoder_hidden_dim / 2, return_sequences=True,
                                             name='query_encoder_lstm')
        else:
            self.query_encoder_lstm = LSTM(config.word_embed_dim, config.encoder_hidden_dim, return_sequences=True,
                                           name='query_encoder_lstm')

        self.decoder_lstm = CondAttLSTM(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                        config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                        name='decoder_lstm')

        self.src_ptr_net = PointerNet()

        self.terminal_gen_softmax = Dense(config.decoder_hidden_dim, 2, activation='softmax', name='terminal_gen_softmax')

        self.rule_embedding_W = initializations.get('normal')((config.rule_num, config.rule_embed_dim), name='rule_embedding_W', scale=0.1)
        self.rule_embedding_b = shared_zeros(config.rule_num, name='rule_embedding_b')

        self.node_embedding = initializations.get('normal')((config.node_num, config.node_embed_dim), name='node_embed', scale=0.1)

        self.vocab_embedding_W = initializations.get('normal')((config.target_vocab_size, config.rule_embed_dim), name='vocab_embedding_W', scale=0.1)
        self.vocab_embedding_b = shared_zeros(config.target_vocab_size, name='vocab_embedding_b')

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_rule = Dense(config.decoder_hidden_dim, config.rule_embed_dim, name='decoder_hidden_state_W_rule')

        # decoder_hidden_dim -> action embed
        self.decoder_hidden_state_W_token= Dense(config.decoder_hidden_dim + config.encoder_hidden_dim, config.rule_embed_dim,
                                                 name='decoder_hidden_state_W_token')

        # self.rule_encoder_lstm.params
        self.params = self.query_embedding.params + self.query_encoder_lstm.params + \
                      self.decoder_lstm.params + self.src_ptr_net.params + self.terminal_gen_softmax.params + \
                      [self.rule_embedding_W, self.rule_embedding_b, self.node_embedding, self.vocab_embedding_W, self.vocab_embedding_b] + \
                      self.decoder_hidden_state_W_rule.params + self.decoder_hidden_state_W_token.params

        self.srng = RandomStreams()