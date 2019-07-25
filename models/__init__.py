from .lenet import LeNet
from .resnet import ResNetMnist
from .RPN.rpn import RPN
from .RPN.config import rpn_cfg


def build_model(opt):
	model_name = opt.model_name
	if model_name == 'lenet':
		return LeNet()
	elif model_name == 'resnet_mnist':
		return ResNetMnist()
	elif model_name == 'rpn':
		return RPN(rpn_cfg.DEMENSION_INPUT)
