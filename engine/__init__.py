from .trainer import do_train_normal
from .trainer_rpn import do_train_rpn
from .trainer_metric import do_train_metric


def do_train(opt):
    model_name = opt.model_name
    if model_name == 'rpn':
        return do_train_rpn
    elif opt.use_triplet == 1:
        return do_train_metric
    else:
        return do_train_normal