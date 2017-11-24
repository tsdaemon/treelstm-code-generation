import torch
from torch.autograd import Variable as Var
import torch.nn.functional as F


def ifcond(cond, x_1, x_2):
    # ensure boolean
    cond = cond.byte().float().unsqueeze(1)
    # check is it Tensor or Variable
    if not hasattr(cond, "backward"):
        cond = Var(cond, requires_grad=False)
    return (cond * x_1) + ((1-cond) * x_2)


def from_list(ls, cuda):
    tensor = torch.FloatTensor(ls)
    if cuda:
        tensor = tensor.cuda()
    return tensor


def zeros(*shape, cuda=False):
    t = torch.FloatTensor(*shape).zero_()
    if cuda:
        t = t.cuda()
    return t


def zeros_like(tensor, cuda):
    if hasattr(tensor, "backward"):
        shape = tensor.data.shape
    else:
        shape = tensor.shape
    return zeros(*shape, cuda=cuda)


