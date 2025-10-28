import os
import cv2
import numpy as np
from ultralytics import YOLO
import config

# Load YOLO pose model once
yolo_model = YOLO("yolov8n-pose.pt")  # or your custom hand keypoint model

def get_fingertip_visibility(img_path, fingertip_indices):
    results = yolo_model(img_path, verbose=False)
    if len(results) == 0 or results[0].keypoints is None:
        return [0] * len(fingertip_indices)

    kpts = results[0].keypoints.xy
    confs = results[0].keypoints.conf
    if kpts is None or len(kpts) == 0:   # ✅ extra guard
        return [0] * len(fingertip_indices)

    keypoints = kpts[0].cpu().numpy()
    confs = confs[0].cpu().numpy()

    visibilities = []
    for idx in fingertip_indices:
        if idx < len(keypoints):
            v = 2 if confs[idx] > 0.5 else 0
        else:
            v = 0
        visibilities.append(v)

    return visibilities


def process_label_file(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path}")
        return

    h, w, _ = img.shape

    with open(label_path, "r") as f:
        values = list(map(float, f.read().strip().split()))

    class_id = values[0]
    bbox = values[1:5]
    keypoints = values[5:]

    fingertip_vis = get_fingertip_visibility(img_path, config.FINGERTIP_INDICES)
    new_values = [class_id] + bbox

    for i, idx in enumerate(config.FINGERTIP_INDICES):
        base = idx * 3
        x_norm = keypoints[base]
        y_norm = keypoints[base + 1]
        v = fingertip_vis[i]
        new_values.extend([x_norm, y_norm, v])

    with open(label_path, "w") as f:
        f.write(" ".join(map(str, new_values)))

def run_preprocessing(label_dir, image_dir):
    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue
        img_path = os.path.join(image_dir, filename.replace(".txt", ".jpg"))
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
        label_path = os.path.join(label_dir, filename)
        process_label_file(img_path, label_path)

if __name__ == "__main__":
    print("[INFO] Preprocessing train set...")
    run_preprocessing(config.TRAIN_LABELS, config.TRAIN_IMAGES)
    print("[INFO] Preprocessing val set...")
    run_preprocessing(config.VAL_LABELS, config.VAL_IMAGES)
    print("[INFO] Done!")
