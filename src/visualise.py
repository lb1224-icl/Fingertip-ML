import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml

# Support both `python src/visualise.py` and `python -m src.visualise`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset.hand_kp_yolo import HandKeypointYOLODataset, generate_heatmaps
else:
    from .dataset.hand_kp_yolo import HandKeypointYOLODataset, generate_heatmaps


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor in [0,1] to HWC numpy."""
    return img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def overlay_heatmap(
    image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5, cmap: str = "jet"
) -> np.ndarray:
    """Overlay a single heatmap onto an image."""
    img_np = tensor_to_numpy(image)
    hm = heatmap.cpu().numpy()
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
    color_hm = plt.get_cmap(cmap)(hm_norm)[..., :3]  # RGBA -> RGB
    overlay = (1 - alpha) * img_np + alpha * color_hm
    overlay = np.clip(overlay, 0, 1)
    return overlay


def rotate_sample(sample: dict, angle_deg: float, sigma: float) -> dict:
    """Rotate image/keypoints around center; regenerate heatmaps."""
    if abs(angle_deg) < 1e-3:
        return sample

    img = sample["image"]
    h, w = img.shape[1:]
    center = (w / 2.0, h / 2.0)

    # torchvision rotates counterclockwise; we match coordinate update accordingly.
    img_rot = TF.rotate(img, angle_deg, interpolation=TF.InterpolationMode.BILINEAR)

    kps = sample["keypoints"].clone()
    angle_rad = -angle_deg * math.pi / 180.0
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    if kps.numel() > 0:
        x = kps[..., 0] - center[0]
        y = kps[..., 1] - center[1]
        kps[..., 0] = x * cos_a - y * sin_a + center[0]
        kps[..., 1] = x * sin_a + y * cos_a + center[1]
        outside = (kps[..., 0] < 0) | (kps[..., 0] >= w) | (kps[..., 1] < 0) | (kps[..., 1] >= h)
        kps[..., 2] = torch.where(outside, torch.zeros_like(kps[..., 2]), kps[..., 2])
        kps = kps.clamp(min=0, max=max(h, w))

    heatmaps = generate_heatmaps(kps, sample["heatmaps"].shape[0], h, w, sigma=sigma)

    # Recompute visibility mask after rotation
    if kps.numel() > 0:
        visibility_mask = (kps[:, :, 2] >= 1).any(dim=0).float()
    else:
        visibility_mask = torch.zeros(sample["heatmaps"].shape[0], dtype=torch.float32)

    return {**sample, "image": img_rot, "keypoints": kps, "heatmaps": heatmaps, "mask": visibility_mask}


def plot_sample(
    sample: dict, kp_index: Optional[int], save_path: Optional[Path] = None, show: bool = True
):
    """
    Visualize one sample:
      - Left: RGB image
      - Right: image with either a single keypoint heatmap or max over all keypoints
    """
    image = sample["image"]
    heatmaps = sample["heatmaps"]
    keypoints = sample["keypoints"]
    mask = sample["mask"]

    if kp_index is None:
        # aggregate across all keypoints for a quick sanity check
        hm = torch.max(heatmaps, dim=0).values
        title = "Max over all keypoints"
    else:
        kp_index = int(kp_index)
        hm = heatmaps[kp_index]
        visible = bool(mask[kp_index] > 0.5)
        title = f"Keypoint {kp_index} ({'vis' if visible else 'not vis'})"

    overlay = overlay_heatmap(image, hm)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(tensor_to_numpy(image))
    axes[0].set_title("Image")
    axes[1].imshow(overlay)
    axes[1].set_title(title)

    # Optionally draw raw keypoints for debugging (use first object if multiple).
    if keypoints.numel() > 0:
        # Use first object to avoid clutter; assumes pixel coordinates.
        pts = keypoints[0].cpu()
        for ax in axes:
            ax.scatter(pts[:, 0], pts[:, 1], s=10, c="lime", marker="x")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def interactive_dataset_view(dataset, start_idx: int, kp_index: Optional[int]):
    """Interactive view: use left/right keys to change dataset index."""
    idx_state = {"idx": max(0, min(start_idx, len(dataset) - 1))}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].axis("off")
    axes[1].axis("off")
    title = axes[1].set_title("")

    img_im = axes[0].imshow(np.zeros((10, 10, 3)), origin="upper")
    overlay_im = axes[1].imshow(np.zeros((10, 10, 3)), origin="upper")
    scatters = []

    def redraw():
        for s in scatters:
            s.remove()
        scatters.clear()

        sample = dataset[idx_state["idx"]]
        image = sample["image"]
        heatmaps = sample["heatmaps"]
        keypoints = sample["keypoints"]
        mask = sample["mask"]

        if kp_index is None:
            hm = torch.max(heatmaps, dim=0).values
            title.set_text(f"Idx {idx_state['idx']} (all keypoints)")
        else:
            hm = heatmaps[kp_index]
            visible = bool(mask[kp_index] > 0.5)
            title.set_text(f"Idx {idx_state['idx']} | KP {kp_index} ({'vis' if visible else 'not vis'})")

        img_np = tensor_to_numpy(image)
        h, w = img_np.shape[:2]
        overlay = overlay_heatmap(image, hm)
        img_im.set_data(img_np)
        overlay_im.set_data(overlay)
        img_im.set_extent((0, w, h, 0))
        overlay_im.set_extent((0, w, h, 0))
        axes[0].set_xlim(0, w)
        axes[0].set_ylim(h, 0)
        axes[1].set_xlim(0, w)
        axes[1].set_ylim(h, 0)

        if keypoints.numel() > 0:
            pts = keypoints[0].cpu()
            for ax in axes:
                scatters.append(ax.scatter(pts[:, 0], pts[:, 1], s=10, c="lime", marker="x"))

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            idx_state["idx"] = (idx_state["idx"] + 1) % len(dataset)
            redraw()
        elif event.key == "left":
            idx_state["idx"] = (idx_state["idx"] - 1) % len(dataset)
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overlay ground-truth heatmaps on images.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--root", type=str, default=None, help="Dataset root (overrides config).")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (train/val).")
    parser.add_argument("--idx", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--kp", type=int, default=-1, help="Keypoint index to show; -1 for aggregate.")
    parser.add_argument("--img-size", type=int, default=None, help="Resize for images/heatmaps.")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian sigma for heatmaps.")
    parser.add_argument("--num-keypoints", type=int, default=None, help="Number of keypoints.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure.")
    parser.add_argument("--color-jitter", action="store_true", help="Enable color jitter to see its effect.")
    parser.add_argument("--augment", action="store_true", help="Use dataset augmentations (hflip).")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate sample by degrees (CCW).")
    args = parser.parse_args()

    # Load config defaults
    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}

    def infer_root_split(cfg_dict):
        root = cfg_dict.get("dataset_root") or cfg_dict.get("path") or "data"
        split = cfg_dict.get("train_split") or "train"
        train_path = cfg_dict.get("train")
        if train_path:
            p = Path(train_path)
            if "images" in p.parts:
                try:
                    root = str(p.parent.parent) 
                    split = p.name
                except Exception:
                    pass
        return root, split

    root_default, split_default = infer_root_split(cfg)
    root = args.root or root_default
    split = args.split or split_default
    num_kp = args.num_keypoints or cfg.get("num_keypoints") or (cfg.get("kpt_shape", [21])[0] if cfg.get("kpt_shape") else 21)
    img_size = args.img_size or cfg.get("img_size", 256)
    sigma = args.sigma or cfg.get("sigma", 1.5)

    dataset = HandKeypointYOLODataset(
        root=root,
        split=split,
        num_keypoints=num_kp,
        img_size=img_size,
        sigma=sigma,
        augment=args.augment,
        color_jitter=args.color_jitter,
    )

    sample = dataset[args.idx]
    if abs(args.rotate_deg) > 1e-3:
        sample = rotate_sample(sample, args.rotate_deg, sigma=sigma)

    kp_index = None if args.kp < 0 else args.kp
    save_path = Path(args.save) if args.save else None
    if save_path is not None:
        plot_sample(sample, kp_index, save_path=save_path, show=False)
    else:
        # Interactive by default: left/right keys cycle dataset index.
        interactive_dataset_view(dataset, start_idx=args.idx, kp_index=kp_index)


if __name__ == "__main__":
    main()
