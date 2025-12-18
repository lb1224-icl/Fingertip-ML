import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset.hand_kp_yolo import HandKeypointYOLODataset, collate_fn
    from model.unet_kp import UNetKP
    import eval as eval_utils
else:
    from .dataset.hand_kp_yolo import HandKeypointYOLODataset, collate_fn
    from .model.unet_kp import UNetKP
    from . import eval as eval_utils


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, K, H, W), mask: (B, K)
    Computes per-keypoint MSE and zeros out keys where mask==0.
    """
    mse = (pred - target) ** 2
    mask = mask.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    mse = mse * mask
    denom = mask.sum() + 1e-6
    return mse.sum() / denom


def build_dataloaders(cfg: Dict):
    train_ds = HandKeypointYOLODataset(
        root=cfg["dataset_root"],
        split=cfg.get("train_split", "train"),
        num_keypoints=cfg["num_keypoints"],
        img_size=cfg["img_size"],
        sigma=cfg.get("sigma", 1.5),
        augment=cfg.get("augment", False),
        color_jitter=cfg.get("color_jitter", False),
    )

    val_loader = None
    try:
        val_ds = HandKeypointYOLODataset(
            root=cfg["dataset_root"],
            split=cfg.get("val_split", "val"),
            num_keypoints=cfg["num_keypoints"],
            img_size=cfg["img_size"],
            sigma=cfg.get("sigma", 1.5),
            augment=False,
            color_jitter=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            collate_fn=collate_fn,
        )
    except Exception as e:
        print(f"[WARN] Validation loader not created: {e}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        heatmaps = batch["heatmaps"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = masked_mse_loss(outputs, heatmaps, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train heatmap-based hand keypoint model.")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = UNetKP(num_keypoints=cfg["num_keypoints"], base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    save_dir = Path(cfg.get("save_dir", "outputs/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_pck = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        metrics = {"pck": None, "mean_pixel_error": None, "num_keypoints": 0}
        if val_loader is not None:
            metrics = eval_utils.evaluate_model(
                model,
                val_loader,
                device,
                pck_threshold=cfg.get("pck_threshold", 5.0),
            )
            print(
                f"Val - PCK@{cfg.get('pck_threshold', 5.0)}: {metrics['pck']:.3f}, "
                f"MPE: {metrics['mean_pixel_error']:.2f} px over {metrics['num_keypoints']} kps"
            )

        print(f"Train loss: {train_loss:.4f}")

        # Save checkpoint every epoch
        ckpt_path = save_dir / f"epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "metrics": metrics,
                "config": cfg,
            },
            ckpt_path,
        )

        # Track best PCK
        if metrics["pck"] is not None and metrics["pck"] > best_pck:
            best_pck = metrics["pck"]
            torch.save(model.state_dict(), save_dir / "best_model.pth")

        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

    # Save training history for logging/proof
    hist_path = save_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {hist_path}")


if __name__ == "__main__":
    main()
