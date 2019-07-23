from .trainer import do_train_normal
from .trainer_rpn import do_train_rpn


def do_train(opt):
	model_name = opt.model_name
	if model_name == 'rpn':
		return do_train_rpn
	else:
		return do_train_normal