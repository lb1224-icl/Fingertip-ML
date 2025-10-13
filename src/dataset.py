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

import config

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
        # --- Load image ---
        img_name = self.train_image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # --- Load label file ---
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, base_name + ".txt")
        with open(label_path, "r") as f:
            values = f.read().strip().split()
        values = list(map(float, values))

        # YOLOv8 keypoint format:
        # class_id (1) + bbox (4) + 21 keypoints × (x, y, v)
        class_id = int(values[0])
        bbox = values[1:5]             # (x_center, y_center, w, h)
        keypoints = values[5:]         # remaining 63 values (21×3)

        fingertip_coords = []
        for idx_key in config.FINGERTIP_INDICES:  # e.g. [4, 8, 12, 16, 20]
            base = idx_key * 3
            x = keypoints[base + 0]
            y = keypoints[base + 1]
            v = keypoints[base + 2]   # visibility (0, 1, or 2)

            fingertip_coords.extend([x, y])

        fingertip_coords = np.array(fingertip_coords, dtype=np.float32)
        
        # --- Apply transforms ---
        if self.transform:
            image = self.transform(image)

        # --- Return ---
        return image, torch.tensor(fingertip_coords, dtype=torch.float32)