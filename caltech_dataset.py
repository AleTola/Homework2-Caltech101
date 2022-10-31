
from PIL import Image

import os
import os.path
import sys

from torchvision.datasets import VisionDataset



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        # root = 'Caltech101/101_ObjectCategories'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.labels = {}
        num_label = 0

        self.list = []

        # tuples (image_path, label) 

        with open("Caltech101/" + split + ".txt") as file_in:
            for line in file_in:
                # label = "accordion"
                line = line.rstrip("\n")
                label = line.split('/')[0]

                if label not in self.labels:
                    self.labels[label] = num_label
                    num_label += 1

                if(label != "BACKGROUND_Google"):
                    # image = "image_0002.jpg"
                    image = line.split('/')[1]
                    self.list.append((line, self.labels[label]))
    

    def __getitem__(self, index):
        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        dir = self.root + "/" + self.list[index][0]

        image = Image.open(dir)
        label = self.list[index][1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        # Provide a way to get the length (number of elements) of the dataset
        length = len(self.list)
        return length
