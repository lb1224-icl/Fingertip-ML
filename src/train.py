import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import FingertipDataset
from model import FingertipResNet
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
])

train_dataset = FingertipDataset(config.TRAIN_IMAGES, config.TRAIN_LABELS, transform)
val_dataset = FingertipDataset(config.VAL_IMAGES, config.VAL_LABELS, transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

model = FingertipResNet(num_outputs=config.NUM_OUTPUTS).to(device)
criterion = nn.MSELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_loss = float("inf")

for epoch in range(config.NUM_EPOCHS):
    # --- TRAIN ---
    model.train()
    total_train_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)

        pred_xy = outputs[:, 0::3]
        true_xy = labels[:, 0::3]
        vis_mask = labels[:, 2::3] == 2
        loss_mat = criterion(pred_xy, true_xy)
        masked_loss = (loss_mat * vis_mask).sum() / (vis_mask.sum() + 1e-8)

        masked_loss.backward()
        optimizer.step()
        total_train_loss += masked_loss.item() * imgs.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # --- VALIDATION ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [VAL]"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            pred_xy = outputs[:, 0::3]
            true_xy = labels[:, 0::3]
            vis_mask = labels[:, 2::3] == 2
            loss_mat = criterion(pred_xy, true_xy)
            masked_loss = (loss_mat * vis_mask).sum() / (vis_mask.sum() + 1e-8)
            total_val_loss += masked_loss.item() * imgs.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    scheduler.step()

    print(f"[EPOCH {epoch+1}/{config.NUM_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- SAVE BEST ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
        print(f"[INFO] ✅ Saved best model at epoch {epoch+1} (val loss {best_val_loss:.4f})")

print(f"[INFO] Training complete. Best val loss: {best_val_loss:.4f}")
