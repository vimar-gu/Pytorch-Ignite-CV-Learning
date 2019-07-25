import torch.nn as nn


def get_loss_fn(opt):
	loss_fn_name = opt.loss_fn
	if loss_fn_name == 'crossentropy':
		return nn.CrossEntropyLoss()
	elif loss_fn_name == 'nll':
		return nn.NLLLoss()
	elif loss_fn_name == 'mse':
		return nn.MSELoss()
	elif loss_fn_name == 'l1':
		return nn.L1Loss()
