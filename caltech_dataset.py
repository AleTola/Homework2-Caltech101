
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

        self.labels = {}
        num_label = 0

        self.list = []

        d = os.getcwd() #Gets the current working directory
        print(d)

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
    

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory

        - PyTorch Dataset classes use indexes to read elements

        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
          
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        dir = self.root + "/" + list[index][0]

        image = Image.open(dir)
        label = self.list[index][1]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        # Provide a way to get the length (number of elements) of the dataset
        length = len(self.list)
        return length
