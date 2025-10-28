import cv2
import numpy as np
from torchvision import transforms
from dataset import FingertipDataset
import config

SCALE_FACTOR = 4
POINT_RADIUS = 8
FONT_SCALE = 0.8
FONT_THICKNESS = 2

transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
])

dataset = FingertipDataset(config.TRAIN_IMAGES, config.TRAIN_LABELS, transform)

def denormalize_image(tensor):
    img = tensor.clone().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    img = (img[:, :, ::-1] * 255).astype(np.uint8)
    return np.ascontiguousarray(img)

def visualize_dataset():
    idx = 0
    total = len(dataset)
    print(f"[INFO] Viewing {total} samples. Use ←/→ to navigate, Enter to quit.")

    while True:
        image_tensor, labels = dataset[idx]
        image = denormalize_image(image_tensor)
        h, w, _ = image.shape
        coords = labels.numpy().reshape(-1, 3)

        image_up = cv2.resize(
            image, (w * SCALE_FACTOR, h * SCALE_FACTOR), interpolation=cv2.INTER_NEAREST
        )

        for i, (x, y, v) in enumerate(coords):
            if v < 2:
                continue  # skip invisible
            px = int(x * w * SCALE_FACTOR)
            py = int(y * h * SCALE_FACTOR)
            cv2.circle(image_up, (px, py), POINT_RADIUS, (0, 0, 255), -1)
            cv2.putText(image_up, str(i), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)

        cv2.putText(image_up, f"Sample {idx+1}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Fingertip Viewer", image_up)

        key = cv2.waitKey(0)
        if key == 83 or key == ord('d'):
            idx = (idx + 1) % total
        elif key == 81 or key == ord('a'):
            idx = (idx - 1) % total
        elif key in [13, 10]:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_dataset()
