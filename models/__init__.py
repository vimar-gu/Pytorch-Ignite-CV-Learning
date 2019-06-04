from .lenet import LeNet


def build_model(opt):
	model_name = opt.model_name
	if model_name == 'lenet':
		return LeNet()
