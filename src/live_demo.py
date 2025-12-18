import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from model.unet_kp import UNetKP
    import eval as eval_utils
else:
    from .model.unet_kp import UNetKP
    from . import eval as eval_utils


def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_model(model_path: Path, num_keypoints: int, device: torch.device) -> UNetKP:
    model = UNetKP(num_keypoints=num_keypoints, base_channels=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_frame(frame_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    return tensor


def draw_predictions(frame_bgr: np.ndarray, preds: np.ndarray, confs: np.ndarray, mid_th: float, high_th: float):
    """Draw predicted keypoints with color-coded confidence."""
    for (x, y), c in zip(preds.astype(int), confs):
        if c >= high_th:
            color = (0, 255, 0)  # green
        elif c >= mid_th:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 0, 255)  # red
        cv2.drawMarker(frame_bgr, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=6, thickness=2)


def main():
    parser = argparse.ArgumentParser(description="Live webcam hand keypoint overlay.")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best_model.pth", help="Path to model (.pth).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file for defaults.")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (0 is default).")
    parser.add_argument("--img-size", type=int, default=None, help="Resize side for model input.")
    parser.add_argument("--num-keypoints", type=int, default=None, help="Override number of keypoints.")
    parser.add_argument("--conf-mid", type=float, default=0.2, help="Confidence threshold for yellow.")
    parser.add_argument("--conf-high", type=float, default=0.5, help="Confidence threshold for green.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    num_kp = args.num_keypoints or cfg.get("num_keypoints") or (cfg.get("kpt_shape", [21])[0] if cfg.get("kpt_shape") else 21)
    img_size = args.img_size or cfg.get("img_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        sys.exit(1)
    model = load_model(model_path, num_kp, device)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.cam}")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame.")
                continue

            inp = preprocess_frame(frame, img_size).unsqueeze(0).to(device)

            with torch.no_grad():
                heatmaps = model(inp)
                preds = eval_utils.heatmaps_to_coords(heatmaps)[0].cpu().numpy()  # (K, 2)
                confs = heatmaps.amax(dim=(2, 3))[0].cpu().numpy()  # (K,)

            # Draw on resized frame for display
            display = cv2.resize(frame, (img_size, img_size))
            draw_predictions(display, preds, confs, mid_th=args.conf_mid, high_th=args.conf_high)

            cv2.imshow("Hand Keypoints (red=pred)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
