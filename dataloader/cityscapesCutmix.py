import os

import torch
import numpy as np
import random

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.scripts import CityLabels

class CityScapes(Dataset):
    def __init__(self, image_path, label_path, transforms, num_mix=1, beta=1.0, prob=1.0):
        self.data_path = image_path
        self.label_path = label_path

        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

        try:
            print("Loading Data and Labels from the numpy files.....")
            self.data = np.load(data_path)
            self.label = np.load(label_path)

            self.label = np.array([self.encode(mask) for mask in self.label])
        except:
            print("Numpy files not found")

        self.transform = transforms

    def __getitem__(self, index):
        image = (self.data[index]).copy()
        label = (self.label[index]).copy()

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))


            image2 = self.data[rand_index]
            label2 = self.label[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.shape, lam)

            image[bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]
            label[bby1:bby2, bbx1:bbx2] = label2[bby1:bby2, bbx1:bbx2]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)

    def encode(self, mask):
        res = np.zeros_like(mask)
        for label in CityLabels:
            res[mask == label.id] = label.trainId
            return res

    def rand_bbox(self, size, lam):
        if len(size) == 4:
            W = size[1]
            H = size[0]
        elif len(size) == 3:
            W = size[1]
            H = size[0]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CityScapesDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test']

        self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(
              mean = [0.5, 0.5, 0.5],
              std =  [0.5, 0.5, 0.5],
              )
        ])

        if self.config.mode == 'train':
            train_set = CityScapes(self.config.train_X,
                            self.config.train_y, 
                            transform=self.transform)
            valid_set = CityScapes(self.config.val_X,
                            self.config.val_y,
                            transform=self.transform)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)

            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = CityScapes(self.config.data_root,
                           transform=self.transform)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
