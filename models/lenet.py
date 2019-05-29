import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
		self.fc1 = nn.Linear(in_features=256, out_features=84)
		self.fc2 = nn.Linear(in_features=84, out_features=10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=-1)
