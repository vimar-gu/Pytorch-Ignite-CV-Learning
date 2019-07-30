import torch
import torch.nn as nn

from utils.model_init import weights_init_kaiming, weights_init_classifier


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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


class ResNetMetric(nn.Module):
	def __init__(self, opt):
		super(ResNetMetric, self).__init__()
		self.in_planes = 512
		self.base = ResNet(last_stride=opt.last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
		self.base.load_param('/home/srtp/.torch/models/resnet50-19c8e357.pth')

		self.gap = nn.AdaptiveAvgPool2d(1)
		self.num_classes = opt.num_classes
		self.with_bnneck = opt.bnneck

		if self.with_bnneck == 1:
			self.bottleneck = nn.BatchNorm1d(self.in_planes)
			self.bottleneck.bias.requires_grad_(False)
			self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

			self.bottleneck.apply(weights_init_kaiming)
			self.classifier.apply(weights_init_classifier)
		else:
			self.classifier = nn.Linear(self.in_planes, self.num_classes)

	def forward(self, x):
		global_feat = self.gap(self.base(x))
		global_feat = torch.flatten(global_feat, 1)

		if self.with_bnneck == 1:
			feat = self.bottleneck(global_feat)
		else:
			feat = global_feat

		if self.training:
			cls_score = self.classifier(feat)
			return cls_score, feat
		else:
			return feat

	def load_param(self, trained_path):
		param_dict = torch.load(trained_path)
		for i in param_dict:
			self.state_dict()[i].copy_(param_dict[i])
