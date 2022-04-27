
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader


class Cifar10():
    def __init__(self, mode="train"):

        self.mode = mode

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        self.transform_valid = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])


        if self.mode == 'train' or self.mode == 'valid':

            self.cifar10 = datasets.CIFAR10('./data', train=True, download=True)

            data_source = self.cifar10.data
            label_source = self.cifar10.targets
            label_source = np.array(label_source)

            self.data = []
            self.labels = []
            classes = range(10)

            ## training data
            if self.mode == 'train':
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[0:4700]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[0:4700]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

            elif self.mode == 'valid': ## validation data

                classes = range(10)
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[4700:5000]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[4700:5000]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

        elif self.mode == 'test':
            self.cifar10 = datasets.CIFAR10('./data', train=False, download=True)
            self.data = self.cifar10.data
            self.labels = self.cifar10.targets


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        img = self.data[index]
        target =  self.labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.mode == 'train':
            img = self.transform(img)
        elif self.mode == 'valid':
            img = self.transform_valid(img)
        elif self.mode == 'test':
            img = self.transform_valid(img)
        return img, target

def get_cifar10_loader(batch_size):

    dataset1 = Cifar10(mode='train')
    dataset2 = Cifar10(mode='valid')
    dataset3 = Cifar10(mode='test')

    train_loader = DataLoader(dataset=dataset1,
                             batch_size=batch_size,
                             shuffle=True)

    valid_loader = DataLoader(dataset=dataset2,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=dataset3,
                              batch_size=batch_size,
                              shuffle=True)

    return train_loader, valid_loader, test_loader
