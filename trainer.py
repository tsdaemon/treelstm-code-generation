import torch
from tqdm import tqdm
import logging
import astor
import os
import numpy as np
import math
import shutil
import pandas as pd

from lang.parse import decode_tree_to_python_ast
from utils.general import get_batches
from utils.eval import evaluate_decode_result
from utils.io import send_telegram


class Trainer(object):
    def __init__(self, model, config, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer

    def train_all(self, train_data, dev_data, test_data, results_dir):
        max_epoch = self.config.max_epoch
        patience_counter = 0
        history_valid_perf = []
        history_valid_bleu = []
        history_valid_acc = []
        history_errors = []
        best_model_file = None
        validation_bleu, validation_accuracy, validation_errors = 0.0, 0.0, 0.0
        for epoch in range(max_epoch):
            mean_loss = self.train(train_data, epoch)
            logging.info('Epoch {} training finished, mean loss: {}.'.format(epoch+1, mean_loss))

            epoch_dir = os.path.join(results_dir, str(epoch+1))
            if os.path.exists(epoch_dir):
                shutil.rmtree(epoch_dir)
            os.mkdir(epoch_dir)
            model_path = os.path.join(epoch_dir, 'model.pth')
            logging.info('Saving model at {}.'.format(model_path))
            torch.save(self.model, model_path)

            bleu, accuracy, errors = self.validate(dev_data, epoch, epoch_dir)
            logging.info('Epoch {} validation finished, bleu: {}, accuracy: {}, errors: {}.'.format(
                epoch + 1, bleu, accuracy, errors))

            history_valid_acc.append(accuracy)
            history_valid_bleu.append(bleu)
            history_errors.append(errors)
            val_perf = eval(self.config.valid_metric)

            if val_perf > 0.2:
                if len(history_valid_perf) == 0 or val_perf > np.array(history_valid_perf).max():
                    patience_counter = 0
                    logging.info('Found best model on epoch {}'.format(epoch+1))
                    best_model_file = model_path
                    validation_accuracy = accuracy
                    validation_bleu = bleu
                    validation_error = errors
                else:
                    patience_counter += 1
                    logging.info('Hitting patience_counter: {}'.format(patience_counter))
                    if patience_counter >= self.config.train_patience:
                        logging.info('Early Stop!')
                        break

            history_valid_perf.append(val_perf)
            # save performance metrics on every step
            hist_df = pd.DataFrame(list(zip(history_valid_bleu, history_valid_acc, history_errors)), columns=['BLEU', 'Accuracy', 'Errors'])
            history_file = os.path.join(results_dir, 'hist.csv')
            hist_df.to_csv(history_file, index=False)

        # test set evaluation
        if best_model_file is not None:
            self.model = torch.load(best_model_file)
            dir = os.path.join(results_dir, 'final')
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
            bleu, accuracy, errors = self.validate(test_data, -100, dir)
            logging.info('Test set evaluation finished, bleu: {}, accuracy: {}, errors: {}.'.format(
                bleu, accuracy, errors))
            report_result = {
                "Test BLEU": bleu,
                "Test accuracy": accuracy,
                "Test error": errors,
                "Validation BLEU": validation_bleu,
                "Validation accuracy": validation_accuracy,
                "Validation error": validation_error,
                "Last epoch": epoch,
                "Mean error": np.mean(history_errors)
            }
            self.report_bot(report_result)

    def train(self, dataset, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_size = self.config.batch_size
        indices = torch.randperm(len(dataset))
        if self.config.cuda:
            indices = indices.cuda()
        total_batches = math.floor(len(indices)/batch_size)+1
        batches = get_batches(indices, batch_size)

        for i, batch in tqdm(enumerate(batches), desc='Training epoch '+str(epoch+1)+'', total=total_batches):
            trees, queries, tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq, \
            tgt_action_seq, tgt_action_seq_type = dataset.get_batch(batch)

            loss = self.model.forward_train(trees, queries, tgt_node_seq, tgt_action_seq, tgt_par_rule_seq, tgt_par_t_seq, tgt_action_seq_type)
            assert loss > 0, "NLL can not be less than zero"

            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            logging.debug('Batch {}, loss {}'.format(i+1, loss[0]))

        return total_loss/len(dataset)

    def validate(self, dataset, epoch, out_dir):
        self.model.eval()
        #self.model.before_eval()
        cum_bleu, cum_acc, cum_oracle_bleu, cum_oracle_acc = 0.0, 0.0, 0.0, 0.0
        errors = 0
        # all_references, all_predictions = [], []

        for idx in tqdm(range(len(dataset)), desc='Testing epoch '+str(epoch+1)+''):
            enc_tree, query, query_tokens, \
            _, _, _, _, _, \
            ref_code, ref_code_tree = dataset[idx]

            cand_list = self.model(enc_tree, query, query_tokens)
            candidats = []
            for cid, cand in enumerate(cand_list[:self.config.beam_size]):
                try:
                    ast_tree = decode_tree_to_python_ast(cand.tree)
                    code = astor.to_source(ast_tree)
                    candidats.append((cid, cand, ast_tree, code))
                except:
                    logging.debug("Exception in converting tree to code:"
                                  "id: {}, beam pos: {}".format(idx, cid))
                    errors += 1

            bleu, oracle_bleu, acc, oracle_acc, \
            refer_tokens_for_bleu, pred_tokens_for_bleu = evaluate_decode_result(
                (enc_tree, query, query_tokens, ref_code_tree, ref_code),
                idx, candidats, out_dir, self.config.dataset)

            cum_bleu += bleu
            cum_oracle_bleu += oracle_bleu
            cum_acc += acc

            # all_references.append([refer_tokens_for_bleu])
            # all_predictions.append(pred_tokens_for_bleu)
        #self.model.after_eval()

        cum_bleu /= len(dataset)
        cum_acc /= len(dataset)
        cum_oracle_bleu /= len(dataset)
        cum_oracle_acc /= len(dataset)
        errors /= len(dataset)*self.config.beam_size

        # logging.info('corpus level bleu: %f',
        #              corpus_bleu(all_references, all_predictions,
        #                          smoothing_function=SmoothingFunction().method3))
        # logging.info('sentence level bleu: %f', cum_bleu)
        # logging.info('accuracy: %f', cum_acc)
        # logging.info('oracle bleu: %f', cum_oracle_bleu)
        # logging.info('oracle accuracy: %f', cum_oracle_acc)

        return cum_bleu, cum_acc, errors

    def visualize(self, dataset, writer):
        self.model.train()
        self.optimizer.zero_grad()
        batch_size = 2
        indices = torch.randperm(len(dataset))
        batch = next(get_batches(indices, batch_size))

        trees, queries, tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq, \
        tgt_action_seq, tgt_action_seq_type = dataset.get_batch(batch)

        loss = self.model.forward_train(trees, queries, tgt_node_seq, tgt_action_seq, tgt_par_rule_seq, tgt_par_t_seq, tgt_action_seq_type)
        assert loss > 0, "NLL can not be less than zero"

        loss.backward()
        writer.add_graph(self.model, loss)

    def report_bot(self, report_dict):
        msg = "Finished experiment with config {}.\n\n"
        msg += "\n".jooin(["{}: {}.".format(k, v) for k, v in report_dict.items()])
        send_telegram(msg)
