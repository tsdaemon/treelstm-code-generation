import numpy as np


class Trainer(object):
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def train(self, train_data, dev_data, test_data):
        train_len = len(train_data)
        train_indexes = np.arange(train_len)

        max_epoch = self.config.max_epoch

        cum_updates = 0
        patience_counter = 0
        history_valid_perf = []
        history_valid_bleu = []
        history_valid_acc = []
        best_model_params = best_model_by_acc = best_model_by_bleu = None

        for epoch in range(max_epoch):
            # train_data_iter.reset()
            # if shuffle:
            np.random.shuffle(train_indexes)

            # epoch begin
            print('Epoch {}'.format(epoch))
            print("="*80)
            cum_nb_examples = 0
            loss = 0.0

            for index in train_indexes:
                examples = dataset.get_examples(batch_ids)
                cur_batch_size = len(examples)

                inputs = dataset.get_prob_func_inputs(batch_ids)

                train_func_outputs = self.model.train_func(*inputs)
                batch_loss = train_func_outputs[0]

                cum_nb_examples += cur_batch_size
                loss += batch_loss * batch_size


                if cum_updates % config.valid_per_batch == 0:
                    logging.info('begin validation')

                    decode_results = decoder.decode_python_dataset(self.model, self.val_data, verbose=False)
                    bleu, accuracy = evaluation.evaluate_decode_results(self.val_data, decode_results, verbose=False)

                    val_perf = eval(config.valid_metric)

                    logging.info('avg. example bleu: %f', bleu)
                    logging.info('accuracy: %f', accuracy)

                    if len(history_valid_acc) == 0 or accuracy > np.array(history_valid_acc).max():
                        best_model_by_acc = self.model.pull_params()
                        # logging.info('current model has best accuracy')
                    history_valid_acc.append(accuracy)

                    if len(history_valid_bleu) == 0 or bleu > np.array(history_valid_bleu).max():
                        best_model_by_bleu = self.model.pull_params()
                        # logging.info('current model has best accuracy')
                    history_valid_bleu.append(bleu)

                    if len(history_valid_perf) == 0 or val_perf > np.array(history_valid_perf).max():
                        best_model_params = self.model.pull_params()
                        patience_counter = 0
                        logging.info('save current best model')
                        self.model.save(os.path.join(config.output_dir, 'model.npz'))
                    else:
                        patience_counter += 1
                        logging.info('hitting patience_counter: %d', patience_counter)
                        if patience_counter >= config.train_patience:
                            logging.info('Early Stop!')
                            early_stop = True
                            break
                    history_valid_perf.append(val_perf)

                if cum_updates % config.save_per_batch == 0:
                    self.model.save(os.path.join(config.output_dir, 'model.iter%d' % cum_updates))

            logging.info('[Epoch %d] cumulative loss = %f, (took %ds)',
                         epoch,
                         loss / cum_nb_examples,
                         time.time() - begin_time)

            if early_stop:
                break

        logging.info('training finished, save the best model')
        np.savez(os.path.join(config.output_dir, 'model.npz'), **best_model_params)

        if config.data_type == 'django' or config.data_type == 'hs':
            logging.info('save the best model by accuracy')
            np.savez(os.path.join(config.output_dir, 'model.best_acc.npz'), **best_model_by_acc)

            logging.info('save the best model by bleu')
            np.savez(os.path.join(config.output_dir, 'model.best_bleu.npz'), **best_model_by_bleu)

