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
        self.coco = COCO('{}/coco_2017/annotations/instances_{}.json'.format(self.root_dir, self.set_name))
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
        anns = self.load_anns(index)
        sample = (img, anns)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = os.path.join(self.root_dir, 'coco_2017', self.set_name, image_info['file_name'])
        img = skimage.io.imread(img_path)
        try:
            assert len(img.shape) == 3
        except:
            img = np.tile(img, (3, 1, 1))
            img = np.transpose(img, (1, 2, 0))
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

            ann = np.zeros((1, 5))
            ann[0, :4] = a['bbox']
            ann[0, 4] = self.coco_labels_inverse[a['category_id']]
            anns = np.append(anns, ann, axis=0)

        anns[:, 2] += anns[:, 0]
        anns[:, 3] += anns[:, 1]

        return anns

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image['width']) / float(image['height'])


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self.std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    def __call__(self, sample):
        image, anns = sample
        return ((image.astype(np.float32) - self.mean) / self.std, anns)


class Resizer():
    def __call__(self, sample, target_size=256):
        image, anns = sample
        rows, cols = image.shape[:2]

        scale_w = target_size / rows
        scale_h = target_size / cols
        image = skimage.transform.resize(image.astype(np.float64),
                                         (int(round(rows * scale_w)),
                                          int(round(cols * scale_h))),
                                         mode='constant')

        anns[:, 0] *= scale_h
        anns[:, 2] *= scale_h
        anns[:, 1] *= scale_w
        anns[:, 3] *= scale_w
        return (torch.from_numpy(image),
                torch.from_numpy(anns))


def my_coco_show(samples):
    image, anns = samples['img'].numpy(), samples['ann'].numpy()
    img_idx = 1

    for img, ann in zip(image, anns):
        ann = ann[ann[:, 4] != -1]
        if ann.shape[0] == 0:
            continue

        # img = np.transpose(img, (1, 2, 0))
        img = img * np.array([[[0.229, 0.224, 0.225]]]) + np.array([[[0.485, 0.456, 0.406]]])

        for i in range(ann.shape[0]):
            p1 = (int(round(ann[i, 0])), int(round(ann[i, 1])))
            p2 = (int(round(ann[i, 2])), int(round(ann[i, 3])))
            cv2.rectangle(img, p1, p2, (255, 0, 0), 2)

        win_name = str(img_idx)
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(win_name, 10, 10)
        cv2.imshow(win_name, img[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_idx += 1


def detection_collate(data):
    imgs = []
    annots = []
    for s in data:
        img, ann = s
        imgs.append(img)
        annots.append(ann)

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    channels = int(imgs[0].shape[2])
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    im_info = torch.Tensor([max_width, max_height, 1.])
    im_info = im_info.unsqueeze(0)
    im_info = im_info.expand(len(imgs), 3)

    max_num_annots = max(annot.shape[0] for annot in annots)
                                                                            
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    return (padded_imgs, im_info, annot_padded)
