import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset.hand_kp_yolo import collate_fn, HandKeypointYOLODataset
else:
    from .dataset.hand_kp_yolo import collate_fn, HandKeypointYOLODataset


def heatmaps_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Convert heatmaps to (x, y) coordinates by argmax.
    heatmaps: (B, K, H, W)
    returns: (B, K, 2)
    """
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    idx = flat.argmax(dim=-1)  # (B, K)
    y = idx // w
    x = idx % w
    coords = torch.stack([x, y], dim=-1).float()
    return coords


def compute_metrics(
    preds: torch.Tensor,
    gts: torch.Tensor,
    vis_mask: torch.Tensor,
    pck_threshold: float,
) -> Dict[str, float]:
    """
    preds: (B, K, 2)
    gts:   (B, K, 2)
    vis_mask: (B, K) with 1 for supervised keypoints
    """
    diff = preds - gts
    dist = torch.sqrt((diff**2).sum(dim=-1))  # (B, K)

    vis = vis_mask > 0.5
    if vis.sum() == 0:
        return {"pck": 0.0, "mean_pixel_error": 0.0, "num_keypoints": 0}

    dist_vis = dist[vis]
    pck = (dist_vis <= pck_threshold).float().mean().item()
    mpe = dist_vis.mean().item()
    return {"pck": pck, "mean_pixel_error": mpe, "num_keypoints": int(vis.sum().item())}


def extract_gt_coords(keypoints_list, num_keypoints: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From a list of keypoints tensors (one per image, shape (num_objects, K, 3)),
    take the first object if present, else zeros. Mask uses visibility >=1.
    Returns coords (B, K, 2) and mask (B, K).
    """
    batch = len(keypoints_list)
    coords = torch.zeros((batch, num_keypoints, 2), device=device)
    mask = torch.zeros((batch, num_keypoints), device=device)
    for i, kp in enumerate(keypoints_list):
        if kp.numel() == 0:
            continue
        first = kp[0].to(device)
        coords[i, :, 0] = first[:, 0]
        coords[i, :, 1] = first[:, 1]
        mask[i] = (first[:, 2] >= 1).float()
    return coords, mask


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pck_threshold: float,
) -> Dict[str, float]:
    model.eval()
    total_pck = 0.0
    total_mpe = 0.0
    total_kp = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        heatmaps_gt = batch["heatmaps"].to(device)
        mask_gt = batch["mask"].to(device)

        outputs = model(images)
        pred_coords = heatmaps_to_coords(outputs)

        gt_coords, vis_mask = extract_gt_coords(batch["keypoints"], outputs.shape[1], device)

        # Use dataset-provided mask for supervision; if multiple hands, we already reduced to first object.
        combined_mask = torch.minimum(vis_mask, mask_gt)

        metrics = compute_metrics(pred_coords, gt_coords, combined_mask, pck_threshold)
        total_pck += metrics["pck"] * metrics["num_keypoints"]
        total_mpe += metrics["mean_pixel_error"] * metrics["num_keypoints"]
        total_kp += metrics["num_keypoints"]

    if total_kp == 0:
        return {"pck": 0.0, "mean_pixel_error": 0.0, "num_keypoints": 0}

    return {
        "pck": total_pck / total_kp,
        "mean_pixel_error": total_mpe / total_kp,
        "num_keypoints": total_kp,
    }
