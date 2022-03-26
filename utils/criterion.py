"""
    Following codebase is the implementation of semantic segmentation task for 
    Crop Production Mapping on the dataset made available by "Radiant MLHub".
	The link to the dataset is https://mlhub.earth/data/umd_mali_crop_type
    Five different models, namely, Fully Convolutional Network with 8, 16 and 32 layers
	and UNet and SegNet architectures are used to carry out the experiments. To evaluate the models, 
	IoU or Jaccard Similarity is measured over the actual and predicted segmented maps.
"""

# Import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cross entropy loss over the actual segmentation map and predicted segmentation map
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight = None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)