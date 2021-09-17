import os

import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.scripts import CityLabels

class CityScapes(Dataset):
    def __init__(self, image_path, label_path, transforms):
        self.data_path = image_path
        self.label_path = label_path

        self.data = np.load(data_path)
        self.label = np.load(label_path)

        self.label = np.array([self.encode(mask) for mask in self.label])

        self.transform = transforms

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            image = self.transform(image)

            return image, label

    def __len__(self):
        return len(self.data)

    def encode(mask):
        res = np.zeros_like(mask)
        for label in CityLabels:
            res[mask == label.id] = label.trainId
            return res



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
            valid_set = CityScapes('val', self.config.val_X,
                            self.config.val_y
                            transform=self.transform)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False)

            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = CityScapes('test', self.config.data_root,
                           transform=self.transform)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
