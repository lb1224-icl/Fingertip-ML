import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Support both `python src/visualise.py` and `python -m src.visualise`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset.hand_kp_yolo import HandKeypointYOLODataset
else:
    from .dataset.hand_kp_yolo import HandKeypointYOLODataset


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


def main():
    parser = argparse.ArgumentParser(description="Overlay ground-truth heatmaps on images.")
    parser.add_argument("--root", type=str, default="data/hand_keypoint_dataset_26k", help="Dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val).")
    parser.add_argument("--idx", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--kp", type=int, default=-1, help="Keypoint index to show; -1 for aggregate.")
    parser.add_argument("--img-size", type=int, default=256, help="Resize for images/heatmaps.")
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian sigma for heatmaps.")
    parser.add_argument("--num-keypoints", type=int, default=21, help="Number of keypoints.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure.")
    args = parser.parse_args()

    dataset = HandKeypointYOLODataset(
        root=args.root,
        split=args.split,
        num_keypoints=args.num_keypoints,
        img_size=args.img_size,
        sigma=args.sigma,
        augment=False,
        color_jitter=False,
    )

    sample = dataset[args.idx]
    kp_index = None if args.kp < 0 else args.kp
    save_path = Path(args.save) if args.save else None
    plot_sample(sample, kp_index, save_path=save_path, show=True)


if __name__ == "__main__":
    main()
