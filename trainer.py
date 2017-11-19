import numpy as np
import torch
from tqdm import tqdm
from torch.autograd import Variable as Var


class Trainer(object):
    def __init__(self, model, config, optimizer, criterion, metrics):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics


    def train_all(self, train_data, dev_data, test_data, results_dir):
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
            self.train(train_data, epoch)



    def train(self, dataset, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(indices, desc='Training epoch '+str(epoch+1)+''):
            enc_tree, dec_tree, input, code = dataset[idx]
            input = Var(input)
            if self.config.cuda:
                input = input.cuda()
            output = self.model(enc_tree, input)
            err = self.criterion(output, dec_tree)
            loss += err.data[0]
            err.backward()
            k += 1
            if k % self.config.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        return loss/len(dataset)

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