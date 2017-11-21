import torch.nn as nn
import torch.functional as F


class SoftmaxDense(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        o = super(SoftmaxDense).forward(self, input)
        return nn.Softmax(o)