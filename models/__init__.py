from .lenet import LeNet
from .resnet import ResNetMnist, ResNetMetric
from .RPN.rpn import RPN
from .RPN.config import rpn_cfg


def build_model(opt):
	model_name = opt.model_name
	if model_name == 'lenet':
		return LeNet(opt)
	elif model_name == 'resnet_metric':
		return ResNetMetric(opt)
	elif model_name == 'resnet_mnist':
		return ResNetMnist(opt)
	elif model_name == 'rpn':
		return RPN(rpn_cfg.DEMENSION_INPUT)
