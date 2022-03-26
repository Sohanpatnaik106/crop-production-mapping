import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CropMappingDataset(Dataset):

    def __init__(self, images_path, labels_path, image_size = (256, 256), image_type = "grayscale"):
        super(Dataset, self).__init__()

        self.images_path = images_path
        self.labels_path = labels_path

        self.dataset_size = len(images_path)

        self.image_size = image_size
        self.image_type = image_type

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        image_path = self.images_path[idx]
        label_path = self.labels_path[idx]

        image = Image.open(image_path)
        image = self.transform(image)

        label = Image.open(label_path)
        label = self.transform(label)

        return image, label