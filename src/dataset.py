import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FingertipDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.endswith((".jpg", ".png"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        label_path = os.path.join(
            self.label_dir, os.path.splitext(img_name)[0] + ".txt"
        )

        with open(label_path, "r") as f:
            values = list(map(float, f.read().strip().split()))

        keypoints = values[5:]  # skip class_id + bbox

        fingertip_data = []
        for i in range(5):
            base = i * 3
            x = keypoints[base]
            y = keypoints[base + 1]
            v = keypoints[base + 2]
            fingertip_data.extend([x, y, v])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(fingertip_data, dtype=torch.float32)
