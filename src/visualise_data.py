import cv2
import numpy as np
from dataset import FingertipDataset
from torchvision import transforms
import config  # stores FINGERTIP_INDICES, paths, etc.

# === SETTINGS ===
SCALE_FACTOR = 4
POINT_RADIUS = 8
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor()
])

# === DATASET ===
dataset = FingertipDataset(
    img_dir=config.DATA_DIR + "/images/train",
    label_dir=config.DATA_DIR + "/labels/train",
    transform=transform
)

# === HELPER: reverse normalization ===
MEAN = config.IMAGENET_MEAN
STD = config.IMAGENET_STD

def denormalize_image(tensor):
    img = tensor.clone().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(STD) + np.array(MEAN)
    img = np.clip(img, 0, 1)
    img = (img[:, :, ::-1] * 255).astype(np.uint8)  # RGB->BGR for OpenCV
    img = np.ascontiguousarray(img)
    return img

# === VISUALIZATION LOOP ===
def visualize_dataset():
    idx = 0
    total = len(dataset)
    print(f"[INFO] Viewing {total} samples. Use ←/→ to navigate, Enter to quit.")

    while True:
        image_tensor, labels = dataset[idx]
        image = denormalize_image(image_tensor)

        h, w, _ = image.shape
        coords = labels.numpy().reshape(-1, 2)

        # Upscale image
        image_up = cv2.resize(
            image, (w * SCALE_FACTOR, h * SCALE_FACTOR), interpolation=cv2.INTER_NEAREST
        )

        # Draw points
        for i, (x, y) in enumerate(coords):
            px = int(x * w * SCALE_FACTOR)
            py = int(y * h * SCALE_FACTOR)
            cv2.circle(image_up, (px, py), POINT_RADIUS, (0, 0, 255), -1)
            cv2.putText(image_up, str(i), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)

        # Add info text
        cv2.putText(
            image_up,
            f"Sample {idx+1}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow("Fingertip Viewer", image_up)

        key = cv2.waitKey(0)

        # Right arrow
        if key == 83 or key == ord('d'):  # Windows/Unix arrow code or fallback
            idx = (idx + 1) % total
        # Left arrow
        elif key == 81 or key == ord('a'):
            idx = (idx - 1) % total
        # Enter (13 is Windows, 10 Unix)
        elif key in [13, 10]:
            print("[INFO] Exiting viewer.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_dataset()
