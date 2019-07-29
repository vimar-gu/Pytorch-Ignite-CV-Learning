import os
import cv2
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Cars196Dataset(Dataset):
	def __init__(self, root_dir, set_name, transform=None):
		super(Cars196Dataset, self).__init__()
		self.root_dir, self.set_name = root_dir, set_name
		self.transform = transform
		self.annotation_mat = loadmat('{}/car_devkit/devkit/cars_{}_annos.mat'.format(self.root_dir, self.set_name))
		self.image_dir = '{}/cars_{}'.format(self.root_dir, self.set_name)
		self.image_list, self.image_dict = self.load_image_dict()

	def load_image_dict(self):
		image_list = []
		image_dict = {}
		for annotation in self.annotation_mat['annotations'][0]:
			image_list.append(annotation[5].item())
			image_dict[annotation[5].item()] = annotation[4].item()
		return image_dict

	def __getitem__(self, index):
		image = cv2.imread(os.path.join(self.image_dir, self.image_list[index]))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.transform is not None:
			image = self.transform(image)

		target = self.image_dict[self.image_list[index]]

		return (image, target)

	def __len__(self):
		return len(self.image_list)
