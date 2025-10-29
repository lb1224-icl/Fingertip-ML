import os
import cv2
import torch
import numpy as np
from torchvision import transforms

from dataset import FingertipDataset
from model import FingertipResNet
import config

# === SETTINGS ===
SCALE_FACTOR = 4
POINT_RADIUS = 8
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
])

# === LOAD DATASET ===
dataset = FingertipDataset(
    img_dir=os.path.join(config.DATA_DIR, "images/val"),
    label_dir=os.path.join(config.DATA_DIR, "labels/val"),
    transform=transform
)

# === LOAD MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FingertipResNet(num_outputs=10, pretrained=False).to(device)

model_name = input("Name of model (ending in .pth): ")

# Replace this filename with the trained model you want to visualise
MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, model_name)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === HELPER: denormalize image ===
def denormalize_image(tensor):
    img = tensor.clone().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    img = (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB→BGR
    return np.ascontiguousarray(img)

# === VISUALIZATION LOOP ===
def visualize_model_predictions():
    idx = 0
    total = len(dataset)
    print(f"[INFO] Viewing {total} samples. ←/→ to navigate, Enter to quit.")

    while True:
        image_tensor, label_tensor = dataset[idx]
        image = denormalize_image(image_tensor)
        h, w, _ = image.shape

        # Get ground truth coords
        gt_coords = label_tensor.numpy().reshape(-1, 2)

        # Get model predictions
        with torch.no_grad():
            preds = model(image_tensor.unsqueeze(0).to(device))
        pred_coords = preds.cpu().numpy().reshape(-1, 2)

        # Upscale image for clearer display
        image_up = cv2.resize(
            image, (w * SCALE_FACTOR, h * SCALE_FACTOR), interpolation=cv2.INTER_NEAREST
        )

        # === Draw Ground Truth (🟩 Green) ===
        for i, (x, y) in enumerate(gt_coords):
            px = int(x * w * SCALE_FACTOR)
            py = int(y * h * SCALE_FACTOR)
            cv2.circle(image_up, (px, py), POINT_RADIUS, (0, 255, 0), -1)
            cv2.putText(image_up, f"GT {i}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)

        # === Draw Predictions (🟥 Red) ===
        for i, (x, y) in enumerate(pred_coords):
            px = int(x * w * SCALE_FACTOR)
            py = int(y * h * SCALE_FACTOR)
            cv2.circle(image_up, (px, py), POINT_RADIUS, (0, 0, 255), -1)
            cv2.putText(image_up, f"P {i}", (px + 10, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

        # Info overlay
        cv2.putText(
            image_up,
            f"Sample {idx+1}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow("Fingertip Model Viewer", image_up)

        key = cv2.waitKey(0)
        if key == 83 or key == ord('d'):  # right arrow / D
            idx = (idx + 1) % total
        elif key == 81 or key == ord('a'):  # left arrow / A
            idx = (idx - 1) % total
        elif key in [13, 10]:  # enter
            print("[INFO] Exiting viewer.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_model_predictions()
