import os
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FingertipDataset, download_files
from model import FingertipResNet
import config

# --- Helper function to compute average pixel error ---
def mean_pixel_error(preds, targets, img_size=config.IMG_SIZE[0]):
    # preds, targets shape: [B, 10]
    diff = torch.abs(preds - targets) * img_size
    return diff.mean().item()


def download_dataset_if_needed():
    if not os.path.exists(config.DATA_DIR):
        print("📥 Downloading dataset...")
        download_files()
    else:
        print("✅ Dataset already exists.")


def train_model(num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on {device}")

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])

    # --- Datasets & Loaders ---
    train_dataset = FingertipDataset(
        img_dir=os.path.join(config.DATA_DIR, "images/train"),
        label_dir=os.path.join(config.DATA_DIR, "labels/train"),
        transform=transform
    )

    val_dataset = FingertipDataset(
        img_dir=os.path.join(config.DATA_DIR, "images/val"),
        label_dir=os.path.join(config.DATA_DIR, "labels/val"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model ---
    model = FingertipResNet(num_outputs=10, pretrained=True).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.STEP_SIZE, config.GAMMA)

    for epoch in range(num_epochs):
        # ======== TRAIN ========
        model.train()
        total_train_loss = 0
        total_train_pixel_error = 0

        train_pbar = tqdm(train_loader, desc=f"🧠 Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)                          # [B, 10]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            total_train_pixel_error += mean_pixel_error(outputs.detach(), labels.detach()) * images.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_train_pixel_error = total_train_pixel_error / len(train_dataset)

        # ======== VALIDATION ========
        model.eval()
        total_val_loss = 0
        total_val_pixel_error = 0

        val_pbar = tqdm(val_loader, desc=f"🔎 Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)

                total_val_loss += val_loss.item() * images.size(0)
                total_val_pixel_error += mean_pixel_error(outputs, labels) * images.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)
        avg_val_pixel_error = total_val_pixel_error / len(val_dataset)

        scheduler.step()

        print(f"📊 Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"| Train PxErr: {avg_train_pixel_error:.2f}px | Val PxErr: {avg_val_pixel_error:.2f}px")

    # --- Save model ---
    timestamp = f"{datetime.now().date()}_{datetime.now().strftime('%H-%M-%S')}"
    save_path = os.path.join(config.MODEL_SAVE_PATH, f"fingertip_model_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved at {save_path}")


if __name__ == "__main__":
    download_dataset_if_needed()
    train_model()
