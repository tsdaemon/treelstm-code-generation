import torch.nn as nn

from model.utils import *


class SoftmaxDense(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_softmax = nn.LogSoftmax()

    def forward(self, input):
        o = super().forward(input)
        return self.log_softmax(o)
