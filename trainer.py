import logging
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

    # helper function for training
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

    # helper function for testing
    def test(self, dataset, epoch):
        self.model.eval()
        loss = 0
        predictions = []
        for idx in tqdm(range(len(dataset)), desc='Testing epoch '+str(epoch)+''):
            enc_tree, dec_tree, input, code = dataset[idx]
            input = Var(input, volatile=True)
            if self.config.cuda:
                input = input.cuda()
            output = self.model(enc_tree, input)
            err = self.criterion(output, dec_tree)
            loss += err.data[0]
            predictions.append(output)
        return loss/len(dataset), predictions