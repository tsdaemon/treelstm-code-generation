import torch


def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)


def zeros(shape, cuda):
    t = torch.FloatTensor(shape)
    if cuda:
        t = t.cuda()
    return t


def zeros_like(tensor, cuda):
    return zeros(tensor.shape, cuda)


