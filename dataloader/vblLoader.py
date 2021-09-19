import os

import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.xview import generate_localization_polygon, generate_damage_polygon
from utils.datasplitting import datasetSplitter

import glob

class VBL(Dataset):
    def __init__(self, data_root, transforms):
        self.data_root = data_root

        self.image_path = self.data_root + "Images/"
        self.label_path = self.data_root + "Labels/"

        self.image_list = []
        self.label_list = []

        print("Appending Images and Labels to the List...")
        for imgPath in sorted(glob.glob(self.image_path + "*.png")):
            image_list.append(imgPath)

        for lblPath in sorted(glob.glob(self.label_path + "*.png")):
            label_list.append(lblPath)

        print("Total Images loaded: ", len(image_list))
        print("Total Labels loaded: ", len(label_list))

        self.transform = transforms

    def __getitem__(self, index):
        imagePath = self.data[index]
        labelPath = self.label[index]

        image = cv2.imread(imagePath)
        label = generate_damage_polygon(labelPath)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)



class VBLDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test']

        self.valid_split = self.config.valid_split
        self.train_split = 1 - self.valid_split

        self.shuffle_dataset = True

        self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(
              mean = [0.5, 0.5, 0.5],
              std =  [0.5, 0.5, 0.5],
              )
        ])

        if self.config.mode == 'train':
            print("---Training Mode---")
            dataset = VBL(self.config.data_root,
                            transforms=self.transform)

            print("Loading Data into DataLoaders...")
            try:
                train_sampler, valid_sampler = datasetSplitter(dataset, self.valid_split, self.shuffle_dataset)

                self.train_loader = DataLoader(dataset, batch_size=self.config.train_batch_size, shuffle=True, sampler=train_sampler)
                self.valid_loader = DataLoader(dataset, batch_size=self.config.valid_batch_size, shuffle=False, sampler=valid_sampler)

                self.train_iterations = (len(dataset * self.train_split) + self.config.train_batch_size) // self.config.train_batch_size
                self.valid_iterations = (len(dataset * self.valid_split) + self.config.valid_batch_size) // self.config.valid_batch_size
                print("Loading Sucessfull")

            except: 
                print("Loading Unsucessfull")

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

        if self.config.mode == 'train':
            print("Length of Train Dataloader: ", len(self.train_loader))
            print("Length of Valid Dataloader: ", len(self.valid_loader))

            

        elif self.config.mode == 'test':
            print("Length of Test Dataloader: ", len(self.test_loader))

        else:
            print("Invalid Mode of Run was provided")
