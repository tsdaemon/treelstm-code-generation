import torch.nn as nn


class SoftmaxDense(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_softmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, input):
        o = super().forward(input)
        return self.softmax(o)

    def forward_train(self, input):
        o = super().forward(input)
        return self.log_softmax(o)
