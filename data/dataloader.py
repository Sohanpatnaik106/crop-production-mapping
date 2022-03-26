"""
    Following codebase is the implementation of semantic segmentation task for 
    Crop Production Mapping on the dataset made available by "Radiant MLHub".
	The link to the dataset is https://mlhub.earth/data/umd_mali_crop_type
    Five different models, namely, Fully Convolutional Network with 8, 16 and 32 layers
	and UNet and SegNet architectures are used to carry out the experiments. To evaluate the models, 
	IoU or Jaccard Similarity is measured over the actual and predicted segmented maps.
"""

# Import the required libraries
import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Custom Dataloader class to create a dataloader of the 
# Crop Production Mapping dataset 
class CropMappingDataset(Dataset):

    def __init__(self, images_path, labels_path, image_size = (256, 256), image_type = "grayscale"):
        super(Dataset, self).__init__()

        self.images_path = images_path
        self.labels_path = labels_path

        self.dataset_size = len(images_path)

        self.image_size = image_size
        self.image_type = image_type

        # Transform converts the PIL image to a torch tensor
        self.transform = transforms.Compose([transforms.ToTensor()])

    # Return the length of the dataset
    def __len__(self):
        return self.dataset_size

    # Return the image and corresponding label given the index in the list
    def __getitem__(self, idx):

        image_path = self.images_path[idx]
        label_path = self.labels_path[idx]

        image = Image.open(image_path)
        image = self.transform(image)
        image = image.repeat(3, 1, 1)

        label = Image.open(label_path)
        label = self.transform(label)

        return image, label