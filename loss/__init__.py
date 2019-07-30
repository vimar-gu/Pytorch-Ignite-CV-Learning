import torch.nn as nn
from .triplet_loss import TripletLoss


def get_loss_fn(opt):
    loss_fn_name = opt.loss_fn
    if loss_fn_name == 'softmax':
        return nn.CrossEntropyLoss()
    elif loss_fn_name == 'nll':
        return nn.NLLLoss()
    elif loss_fn_name == 'mse':
        return nn.MSELoss()
    elif loss_fn_name == 'l1':
        return nn.L1Loss()
    elif loss_fn_name == 'softmax+triplet':
        def loss_func(feat, score, target):
            return nn.CrossEntropyLoss()(score, target) + TripletLoss(opt.triplet_margin)(feat, target)[0]
        return loss_func
