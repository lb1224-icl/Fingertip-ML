import numpy as np
import pandas as pd

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchmetrics

from model import FingertipCNN
from dataset import FingertipDataset, download_files
from torchvision import transforms
import os

from datetime import datetime

DATA_DIR = "../data/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k"
MODEL_SAVE_PATH = f"../modles/fingertip_model_{datetime.now().date()}_{datetime.now().time()}.pth"

def download_dataset_if_needed():
    if not os.path.exists(DATA_DIR):
        print("Downloading dataset...")
        download_files()
    else:
        print("Dataset already exists ")

def train_model(num_epochs = 20, batch_size = 32, lr = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = FingertipDataset(
        img_dir=DATA_DIR + "/images/train",
        label_dir=DATA_DIR + "/labels/train",
        transform=transform
    )

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = FingertipCNN(3).to(device)
    criterion = nn.MSELoss() # Used to evaluate current progress
    optimizer = optim.Adam(model.parameters(), lr=lr) # Improves parameters

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training complete. Model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    download_dataset_if_needed()
    train_model()

