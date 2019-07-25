import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNetMnist(ResNet):
	def __init__(self):
		super(ResNetMnist, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
		self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

	def forward(self, x):
		x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		feat = torch.flatten(x, 1)
		x = self.fc(feat)

		return feat, x
