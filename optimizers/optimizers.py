import torch

def init_optim(optim, params, lr, weight_decay):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.95)
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported Optimizer: {}".format(optim))