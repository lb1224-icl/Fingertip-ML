import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


def _read_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def _parse_yolo_line(
    line: str, num_keypoints: int, img_w: int, img_h: int
) -> Optional[torch.Tensor]:

    parts = line.strip().split()
    if len(parts) < 5 + 3 * num_keypoints:
        return None

    kp_vals: List[float] = []
    # Skip bbox/class (first 5 entries) and parse the remaining triplets.
    for i in range(num_keypoints):
        base = 5 + 3 * i
        x = float(parts[base]) * img_w
        y = float(parts[base + 1]) * img_h
        v = float(parts[base + 2])  # 0: not labeled, 1: labeled but not visible, 2: visible
        kp_vals.extend([x, y, v])

    return torch.tensor(kp_vals, dtype=torch.float32).view(num_keypoints, 3)


def load_yolo_keypoints(
    label_path: Path, num_keypoints: int, img_w: int, img_h: int
) -> torch.Tensor:
    """
    Loads all keypoints in an image. If multiple hands are present, the heatmaps
    will later be merged by taking the per-pixel maximum for each keypoint index.
    """
    if not label_path.exists():
        return torch.zeros((0, num_keypoints, 3), dtype=torch.float32)

    keypoints: List[torch.Tensor] = []
    with label_path.open("r") as f:
        for line in f:
            kp = _parse_yolo_line(line, num_keypoints, img_w, img_h)
            if kp is not None:
                keypoints.append(kp)

    if not keypoints:
        return torch.zeros((0, num_keypoints, 3), dtype=torch.float32)

    return torch.stack(keypoints, dim=0)


def generate_heatmaps(
    keypoints: torch.Tensor,
    num_keypoints: int,
    height: int,
    width: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    keypoints: Tensor shaped (num_objects, num_keypoints, 3) with (x, y, v) in pixel space.
    Returns heatmaps shaped (num_keypoints, height, width).
    """
    device = keypoints.device
    heatmaps = torch.zeros((num_keypoints, height, width), device=device)
    if keypoints.numel() == 0:
        return heatmaps

    y_grid = torch.arange(height, device=device).view(height, 1).float()
    x_grid = torch.arange(width, device=device).view(1, width).float()

    for kp_idx in range(num_keypoints):
        # Accumulate contributions from each object for the same keypoint index.
        per_kp = torch.zeros((height, width), device=device)
        for obj_idx in range(keypoints.shape[0]):
            x, y, v = keypoints[obj_idx, kp_idx]
            if v < 1:  # treat visibility==0 as not supervised
                continue
            g = torch.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma**2))
            per_kp = torch.maximum(per_kp, g)
        heatmaps[kp_idx] = per_kp

    return heatmaps


class HandKeypointYOLODataset(Dataset):
    """
    Minimal dataset for YOLO-format keypoint labels.

    - Expects images in {root}/images/{split}/
    - Expects labels in {root}/labels/{split}/ with identical stem names.
    - Returns image tensor (C, H, W), heatmaps (K, H, W), mask (K,) and raw keypoints.
    """

    def __init__(
        self,
        root: str,
        split: str,
        num_keypoints: int = 21,
        img_size: int = 256,
        sigma: float = 1.5,
        augment: bool = False,
        color_jitter: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.num_keypoints = num_keypoints
        self.img_size = img_size
        self.sigma = sigma
        self.augment = augment
        self.color_aug = (
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if color_jitter
            else None
        )

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_paths = sorted(self.image_dir.glob("*.jpg")) 
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        img = _read_image(img_path)
        orig_w, orig_h = img.size
        keypoints = load_yolo_keypoints(label_path, self.num_keypoints, orig_w, orig_h)

        # Optional flip augmentation (consistent for image + keypoints).
        if self.augment and torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            if keypoints.numel() > 0:
                keypoints[..., 0] = orig_w - keypoints[..., 0]  # x becomes W - x

        # Color jitter (doesn't affect keypoints)
        if self.color_aug is not None:
            img = self.color_aug(img)

        # Resize image and scale keypoints accordingly.
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        if keypoints.numel() > 0:
            keypoints[..., 0] = keypoints[..., 0] * scale_x
            keypoints[..., 1] = keypoints[..., 1] * scale_y
            keypoints = keypoints.clamp(min=0, max=self.img_size - 1e-3)

        # Build heatmaps and visibility mask.
        heatmaps = generate_heatmaps(keypoints, self.num_keypoints, self.img_size, self.img_size, self.sigma)
        visibility_mask = (
            keypoints[:, :, 2] >= 1 if keypoints.numel() > 0 else torch.zeros((0, self.num_keypoints))
        )
        # Collapse multi-object visibilities to a single mask per keypoint via any().
        if visibility_mask.numel() > 0:
            visibility_mask = visibility_mask.any(dim=0).float()
        else:
            visibility_mask = torch.zeros(self.num_keypoints, dtype=torch.float32)

        img_tensor = TF.to_tensor(img)

        return {
            "image": img_tensor,
            "heatmaps": heatmaps,
            "mask": visibility_mask,
            "keypoints": keypoints,  # shape: (num_objects, K, 3)
            "path": str(img_path),
        }


def collate_fn(batch: Sequence[dict]):
    """Simple collate function to stack tensors and keep metadata."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    heatmaps = torch.stack([b["heatmaps"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    keypoints = [b["keypoints"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"image": images, "heatmaps": heatmaps, "mask": masks, "keypoints": keypoints, "path": paths}
