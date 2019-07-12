import os
import cv2
import torch
import skimage
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SimpleCocoDataset(Dataset):
	def __init__(self, root_dir, set_name, transform=None):
		super(SimpleCocoDataset, self).__init__()
		self.root_dir, self.set_name = root_dir, set_name
		self.transform = transform
		self.coco = COCO('{}/annotations/instances_{}.json'.format(self.root_dir, self.set_name))
		self.image_ids = self.coco.getImgIds()
		self.load_classes()

	def load_classes(self):
		categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        ann = self.load_anns(index)
        sample = {'img': img, 'ann': ann}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(img_path)
        return img.astype(np.float32) / 255.0

    def load_anns(self, index):
        annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        anns = np.zeros((0, 5))

        if len(annotation_ids) == 0:
            return anns

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
