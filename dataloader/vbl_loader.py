import os

import torch
import numpy as np
import cv2
import random

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.xview import generate_localization_polygon, generate_damage_polygon
from utils.datasplitting import datasetSplitter

from utils.cutmix import rand_bbox

import glob

class VBL(Dataset):
    def __init__(self, data_root, transforms, cutmix):
        self.data_root = data_root

        self.image_path = self.data_root + "Images/"
        self.label_path = self.data_root + "Labels/"

        self.image_list = []
        self.label_list = []

        # For CutMix Data Augmentation
        self.cutmix = cutmix
        self.num_mix = 5
        self.beta = 1.0
        self.prob = 1.0

        print("Appending Images and Labels to the List...")
        for imgPath in sorted(glob.glob(self.image_path + "*.png")):
            self.image_list.append(imgPath)

        for lblPath in sorted(glob.glob(self.label_path + "*.json")):
            self.label_list.append(lblPath)

        print("Total Images loaded: ", len(self.image_list))
        print("Total Labels loaded: ", len(self.label_list))

        self.transform = transforms

    def __getitem__(self, index):
        imagePath = self.image_list[index]
        labelPath = self.label_list[index]

        image = cv2.imread(imagePath)
        label = generate_damage_polygon(labelPath)

        if self.cutmix:
            for _ in range(self.num_mix):
                r = np.random.rand(1)
                if self.beta <= 0 or r > self.prob:
                    continue
                lam = np.random.beta(self.beta, self.beta)
                rand_index = random.choice(range(len(self)))
                imagePath2 = self.image_list[rand_index]
                labelPath2 = self.label_list[rand_index]
                image2 = cv2.imread(imagePath2)
                label2 = generate_damage_polygon(labelPath2)
                bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)
                image[bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]
                label[bby1:bby2, bbx1:bbx2] = label2[bby1:bby2, bbx1:bbx2]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.label_list)



class VBLDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test']

        self.valid_split = self.config.valid_split
        self.train_split = 1 - self.valid_split

        self.shuffle_dataset = False
        # Some shuffle related error: https://stackoverflow.com/questions/61033726/valueerror-sampler-option-is-mutually-exclusive-with-shuffle-pytorch

        self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(
              mean = [0.5, 0.5, 0.5],
              std =  [0.5, 0.5, 0.5],
              )
        ])

        if self.config.mode == 'train':
            print("---Training Mode---")
            self.dataset = VBL(self.config.data_root,
                            transforms=self.transform,
                            cutmix=self.config.cutmix)

            print("Loading Data into DataLoaders...")

            train_sampler, valid_sampler = datasetSplitter(self.dataset, self.valid_split, self.shuffle_dataset)

            self.train_loader = DataLoader(self.dataset, batch_size=self.config.train_batch_size, shuffle=False, sampler=train_sampler)
            self.valid_loader = DataLoader(self.dataset, batch_size=self.config.valid_batch_size, shuffle=False, sampler=valid_sampler)

            print("Length of Train Loader: ", len(self.train_loader))
            print("Length of Valid Loader: ", len(self.valid_loader))

            self.train_iterations = (len(self.dataset) * self.train_split) // self.config.train_batch_size + 1
            self.valid_iterations = (len(self.dataset) * self.valid_split) // self.config.valid_batch_size + 1


            # Calculating number of pixels of each class type in train and val dataset
            self.train_pixels = np.zeros(5)
            self.valid_pixels = np.zeros(5)

            for batch in self.train_loader: 
                image, label = batch
                unique, counts = np.unique(label, return_counts=True)
                mapping = dict(zip(unique, counts))
                for key in mapping:
                    self.train_pixels[key] += mapping[key]

            for batch in self.valid_loader: 
                image, label = batch
                unique, counts = np.unique(label, return_counts=True)
                mapping = dict(zip(unique, counts))
                for key in mapping:
                    self.valid_pixels[key] += mapping[key]

            self.train_wts = self.train_pixels / np.sum(self.train_pixels)
            self.valid_wts = self.valid_pixels / np.sum(self.valid_pixels)

            print("Class distribution in Train Loader: ", self.train_wts)
            print("Class distribution in Valid Loader: ", self.valid_wts)

        elif self.config.mode == 'test':
            print("---Testing Mode---")
            test_set = VBL(self.config.data_root,
                            transforms=self.transform)

            self.test_loader = DataLoader(test_set, batch_size=self.config.test_batch_size, shuffle=False)
            self.test_iterations = (len(test_set) + self.config.test_batch_size) // self.config.test_batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        print("DataLoader Finalized")
