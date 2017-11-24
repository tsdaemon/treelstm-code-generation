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


def index_select_if_none(input, dim, index, ifnone):
    input_max = input.data.shape[dim]
    index_mask = ((index > 0) * (index < input_max)).eq(0)
    index.masked_fill_(index_mask, input_max)
    input = torch.cat([input, ifnone], dim=0)
    return input[index]


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


def add_padding_and_stack(tensors, cuda, dim=0, max_length=None):
    if max_length is None:
        max_length = max([t.data.shape[dim] for t in tensors])

    result = []
    for tensor in tensors:
        sh = tensor.data.shape
        sh[dim] = max_length-sh[dim]
        assert sh[dim] > 0

        padding = Var(zeros(*sh, cuda=cuda))
        tensor = torch.cat([tensor, padding], dim=dim)
        result.append(tensor)

    return torch.stack(result)



