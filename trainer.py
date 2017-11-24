import torch
from tqdm import tqdm
from torch.autograd import Variable as Var
import logging

from utils.general import get_batches


class Trainer(object):
    def __init__(self, model, config, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer

    def train_all(self, train_data, dev_data, test_data, results_dir):
        max_epoch = self.config.max_epoch

        cum_updates = 0
        patience_counter = 0
        history_valid_perf = []
        history_valid_bleu = []
        history_valid_acc = []
        best_model_params = best_model_by_acc = best_model_by_bleu = None
        for epoch in range(max_epoch):
            mean_loss = self.train(train_data, epoch)
            logging.info('\nEpoch {} training finished, mean loss: {}.'.format(epoch+1, mean_loss))

    def train(self, dataset, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_size = self.config.batch_size
        indices = torch.randperm(len(dataset))
        total_batches = len(indices)/batch_size
        batches = get_batches(indices, batch_size)

        for i, batch in tqdm(enumerate(batches), desc='Training epoch '+str(epoch+1)+'', total=total_batches):
            trees, queries, tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq, \
            tgt_action_seq, tgt_action_seq_type, _ = dataset.get_batch(batch)

            loss = self.model.forward_train(trees, queries, tgt_node_seq, tgt_action_seq, tgt_par_rule_seq, tgt_par_t_seq, tgt_action_seq_type)
            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            logging.debug('Batch {}, loss {}'.format(i+1, loss[0]))

        return total_loss/len(dataset)

    def validate(self, dataset, epoch):
        self.model.eval()
        loss = 0
        predictions = []
        for idx in tqdm(range(len(dataset)), desc='Testing epoch '+str(epoch)+''):
            enc_tree, dec_tree, input, code = dataset[idx]
            input = Var(input, volatile=True)
            if self.config.cuda:
                input = input.cuda()
            output = self.model(enc_tree, input)

            decode_results = decoder.decode_python_dataset(self.model, self.val_data, verbose=False)
            bleu, accuracy = evaluation.evaluate_decode_results(self.val_data, decode_results, verbose=False)

            err = self.criterion(output, dec_tree)
            loss += err.data[0]
            predictions.append(output)
        return loss/len(dataset), predictions