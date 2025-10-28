import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FingertipDataset, download_files
from model import FingertipResNet
from torchvision import transforms
import config
import os
from datetime import datetime

def download_dataset_if_needed():
    if not os.path.exists(config.DATA_DIR):
        print("Downloading dataset...")
        download_files()
    else:
        print("Dataset already exists.")

def train_model(num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # === transforms ===
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])

    # === datasets and loaders ===
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

    # === model ===
    model = FingertipResNet(num_outputs=10, pretrained=True).to(device)

    # Simple, robust loss
    criterion = nn.SmoothL1Loss()   # or nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.STEP_SIZE, config.GAMMA)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)       # shape [B, 10]
            loss = criterion(outputs, labels)  # shape [B, 10] → scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)

        # === validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item() * images.size(0)

        avg_val_loss = total_val_loss / len(val_dataset)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # === save model ===
    timestamp = f"{datetime.now().date()}_{datetime.now().strftime('%H-%M-%S')}"
    save_path = os.path.join(config.MODEL_SAVE_PATH, f"fingertip_model_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved at {save_path}")

if __name__ == "__main__":
    download_dataset_if_needed()
    train_model()
