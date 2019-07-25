import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNetMnist(ResNet):
	def __init__(self):
		super(ResNetMnist, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
