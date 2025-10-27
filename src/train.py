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

from model import FingertipCNN, FingertipResNet
from dataset import FingertipDataset, download_files
from torchvision import transforms
import os

from datetime import datetime

import config



def download_dataset_if_needed():
    if not os.path.exists(config.DATA_DIR):
        print("Downloading dataset...")
        download_files()
    else:
        print("Dataset already exists ")

def train_model(num_epochs = config.NUM_EPOCHS, batch_size = config.BATCH_SIZE, lr = config.LEARNING_RATE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])

    train_dataset = FingertipDataset(
        img_dir=config.DATA_DIR + "/images/train",
        label_dir=config.DATA_DIR + "/labels/train",
        transform=transform
    )

    val_dataset = FingertipDataset(
        img_dir=config.DATA_DIR + "/images/val",
        label_dir=config.DATA_DIR + "/labels/val",
        transform=transform
    ) 

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4,  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 4,  pin_memory=True)

    model = FingertipResNet(num_outputs=10, pretrained=True).to(device)
    criterion = nn.MSELoss() # Used to evaluate current progress
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE) # Improves parameters
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)
  
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH + f"{datetime.now().date()}_{datetime.now().time()}.pth")
    print(f"✅ Training complete. Model saved at {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    download_dataset_if_needed()
    train_model(config.NUM_EPOCHS, config.BATCH_SIZE, config.LEARNING_RATE)

