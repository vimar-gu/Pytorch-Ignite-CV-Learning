import cv2
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from .coco import SimpleCocoDataset, Normalizer, Resizer, detection_collate
from models.RPN.config import rpn_cfg
from torch.utils.data import DataLoader


def get_dataloader(opt):
	dataset_name = opt.dataset_name
	dataset_root = opt.dataset_root
	if dataset_name == 'mnist':
		data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
		train_set = MNIST(download=True, root=dataset_root, transform=data_transform, train=True)
		test_set = MNIST(download=True, root=dataset_root, transform=data_transform, train=False)
		train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
		test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)
	elif dataset_name == 'cifar':
		data_transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		train_set = CIFAR10(download=True, root=dataset_root, transform=data_transform, train=True)
		test_set = CIFAR10(download=True, root=dataset_root, transform=data_transform, train=False)
		train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
		test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)
	elif dataset_name == 'coco':
		data_transform = Compose([Normalizer(), Resizer()])
		train_set = SimpleCocoDataset(root_dir=dataset_root, set_name='train2017', transform = data_transform)
		test_set = SimpleCocoDataset(root_dir=dataset_root, set_name='val2017', transform = data_transform)
		train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, collate_fn=detection_collate)
		test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, collate_fn=detection_collate)
	return (train_loader, test_loader)
