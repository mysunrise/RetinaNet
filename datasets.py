import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from dataencoder import DataEncoder
from utils import resize
import matplotlib.pyplot as plt


def default_loader(path):
    return Image.open(path).convert("RGB")


class VocDataset(data.DataLoader):
    def __init__(self, root, annotation_file, input_size, train=True, transform=None,
                 target_transform=None, loader=default_loader):
        # type: (object, object, object, object, object, object, object) -> object
        self.root = root
        self.annotation_file = annotation_file
        self.input_size = input_size
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.encoder = DataEncoder()

        self.imgs = []
        self.labels = []
        self.bboxes = []

        with open(self.annotation_file) as f:
            lines = f.readlines()
            for line in lines:
                annotation = line.strip().split()
                self.imgs.append(annotation[0])
                bboxes_num = (len(annotation) - 1) / 5
                bbox = []
                label = []
                for i in xrange(bboxes_num):
                    bbox.append([float(annotation[i * 5 + 1]), float(annotation[i * 5 + 2]),
                                 float(annotation[i * 5 + 3]), float(annotation[i * 5 + 4])])
                    label.append(int(annotation[i * 5 + 5]))

                self.bboxes.append(torch.FloatTensor(bbox))
                self.labels.append(torch.LongTensor(label))

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = self.loader(os.path.join(self.root, img_name))
        bboxes = self.bboxes[index].clone()
        labels = self.labels[index]
        size = self.input_size
        img, bboxes = resize(img, bboxes, size)

        # data_argumentation

        if self.transform:
            img = self.transform(img)

        return img, bboxes, labels

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        bboxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        w, h = self.input_size
        imgs_num = len(imgs)

        input = torch.zeros(imgs_num, 3, h, w)
        target_locs = []
        target_clses = []
        for i in xrange(imgs_num):
            input[i] = imgs[i]
            target_loc, target_cls = self.encoder.encode(bboxes[i], labels[i], self.input_size)
            target_locs.append(target_loc)
            target_clses.append(target_cls)

        return input, torch.stack(target_locs), torch.stack(target_clses)

    def __len__(self):
        return len(self.imgs)
