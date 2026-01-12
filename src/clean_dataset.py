import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


def _find_image(images_root: Path, stem: str) -> Path:
    # Try common extensions to locate the image file for a given label stem.
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = images_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return Path()


def _compute_visibility(x: float, y: float, w: int, h: int) -> int:
    # Binary visibility: 0 if out of frame, 2 if inside.
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0
    return 2


def _recompute_bbox(kps: List[Tuple[float, float, int]], w: int, h: int):
    # Recompute a tight bounding box around visible keypoints.
    visible = [(x, y) for x, y, v in kps if v > 0]
    if not visible:
        return None
    xs = [x for x, _ in visible]
    ys = [y for _, y in visible]
    x_min = max(min(xs), 0.0)
    y_min = max(min(ys), 0.0)
    x_max = min(max(xs), w - 1)
    y_max = min(max(ys), h - 1)
    bw = max(x_max - x_min, 1.0)
    bh = max(y_max - y_min, 1.0)
    cx = x_min + bw / 2.0
    cy = y_min + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def clean_labels(dataset_root: Path):
    # Iterate over YOLO label files and fix bbox + visibility flags.
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"
    if not labels_root.exists() or not images_root.exists():
        print("[WARN] images/labels not found; skipping cleaning.")
        return

    label_files = list(labels_root.rglob("*.txt"))
    if not label_files:
        print("[WARN] No label files found; skipping cleaning.")
        return

    print(f"Cleaning {len(label_files)} label files...")
    for label_path in label_files:
        split = label_path.parent.name
        img_dir = images_root / split
        stem = label_path.stem
        img_path = _find_image(img_dir, stem)
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        new_lines = []
        with label_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6: # no keypoints
                    continue
                cls = parts[0]
                kp_vals = parts[5:] # skip bounding box
                if len(kp_vals) % 3 != 0:
                    continue
                num_kp = len(kp_vals) // 3
                kps = []
                for i in range(num_kp):
                    x = float(kp_vals[3 * i]) * w
                    y = float(kp_vals[3 * i + 1]) * h
                    v = _compute_visibility(x, y, w, h)
                    kps.append((x, y, v))

                bbox = _recompute_bbox(kps, w, h)
                if bbox is None:
                    # fallback to original bbox if no visible points
                    cx, cy, bw, bh = map(float, parts[1:5])
                else:
                    cx, cy, bw, bh = bbox

                out_parts = [cls, f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                for x, y, v in kps:
                    out_parts.extend([f"{x / w:.6f}", f"{y / h:.6f}", str(v)])
                new_lines.append(" ".join(out_parts))

        if new_lines:
            label_path.write_text("\n".join(new_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Clean YOLO keypoint labels in-place.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root containing images/ and labels/.")
    args = parser.parse_args()

    clean_labels(Path(args.root))
    print("Done. Labels cleaned in-place.")


if __name__ == "__main__":
    main()
