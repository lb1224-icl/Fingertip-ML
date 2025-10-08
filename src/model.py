import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

class FingertipCNN(nn.Module):
    def __init__(self, in_channels):

        super().__init__()

        # Convolution layers — extract feature maps
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Pooling — downsample feature maps
        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layers to understand feature maps
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10) # 5 fingertips x 2 coords

    # X is the Tensor (input data)
    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x)) # Fully connected layers to understand features
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No ReLU as this is final output layer

        return x



