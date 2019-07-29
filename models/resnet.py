import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNetMnist(ResNet):
	def __init__(self, opt):
		super(ResNetMnist, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		if opt.bnneck == 1:
			self.bottleneck = nn.BatchNorm1d(512)
			self.bottleneck.bias.requires_grad_(False)

	def forward(self, x):
		x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		feat = torch.flatten(x, 1)
		if self.bottleneck is not None:
			out = self.bottleneck(feat)
		out = self.fc(out)

		return feat, out
