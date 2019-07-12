from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from .coco import SimpleCocoDataset


def get_dataset(opt):
	dataset_name = opt.dataset_name
	dataset_root = opt.dataset_root
	if dataset_name == 'mnist':
		data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
		train_set = MNIST(download=True, root=dataset_root, transform=data_transform, train=True)
		test_set = MNIST(download=True, root=dataset_root, transform=data_transform, train=False)
	elif dataset_name == 'cifar':
		data_transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		train_set = CIFAR10(download=True, root=dataset_root, transform=data_transform, train=True)
		test_set = CIFAR10(download=True, root=dataset_root, transform=data_transform, train=False)
	elif dataset_name == 'coco':
		data_transform = Compose([ToTensor()])
		train_set = SimpleCocoDataset(root_dir=dataset_root, set_name='train2017', transform = data_transform)
		train_set = SimpleCocoDataset(root_dir=dataset_root, set_name='val2017', transform = data_transform)
	return (train_set, test_set)
