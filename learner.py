import logging
import torch
from tqdm import tqdm
from torch.autograd import Variable as Var


class Learner(object):
    def __init__(self, model, config, optimizer, criterion, metrics):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)), desc='Training epoch '+str(self.epoch+1)+''):
            enc_tree, dec_tree, input, code = dataset[indices[idx]]
            input = Var(input)
            if self.args.cuda:
                input = input.cuda()
            output = self.model(enc_tree, input)
            err = self.criterion(output, dec_tree)
            loss += err.data[0]
            err.backward()
            k += 1
            if k%self.args.batchsize==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.arange(1,dataset.num_classes+1)
        for idx in tqdm(range(len(dataset)), desc='Testing epoch '+str(self.epoch)+''):
            ltree, lsent, rtree, rsent, label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput)
            err = self.criterion(output, target)
            loss += err.data[0]
            output = output.data.squeeze().cpu()
            predictions[idx] = torch.dot(indices, torch.exp(output))
        return loss/len(dataset), predictions

    def train(self, config):
        dataset = self.train_data
        nb_epoch = config.max_epoch

        logging.info('begin training')
        cum_updates = 0
        patience_counter = 0
        early_stop = False
        history_valid_perf = []
        history_valid_bleu = []
        history_valid_acc = []
        best_model_params = best_model_by_acc = best_model_by_bleu = None