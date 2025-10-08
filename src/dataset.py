import kaggle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm

from PIL import Image
import numpy as np

# Fingertip keypoint indices
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

def download_files():
    kaggle.api.authenticate()

    kaggle.api.dataset_download_files("riondsilva21/hand-keypoint-dataset-26k", path="../data", unzip = True)

class FingertipDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.train_image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    

    def __len__(self):
        return len(self.train_image_files)
    
    def __getitem__(self, idx):
        img_name = self.train_image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, base_name + ".txt")
        with open(label_path, "r") as f:
            values = f.read().strip().split()

        values = list(map(float, values[1:]))

        fingertip_coords = []
        for idx_key in FINGERTIP_INDICES:
            x = values[idx_key * 3]     # x
            y = values[idx_key * 3 + 1] # y
            fingertip_coords.extend([x, y])

        fingertip_coords = np.array(fingertip_coords, dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(fingertip_coords, dtype=torch.float32)