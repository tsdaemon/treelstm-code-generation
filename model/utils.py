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


def softmax(input, dim=1):
    input_size = input.size()

    trans_input = input.transpose(dim, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(dim, len(input_size)-1)


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


