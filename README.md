# Hand Keypoint Heatmaps (PyTorch)

Lightweight U-Net that predicts heatmap targets for 21 hand keypoints (YOLO-format). Includes data loading, training, evaluation (PCK/pixel error), visualization, checkpointing, and a live webcam demo.

## Features
- YOLO keypoint dataset loader with Gaussian heatmap targets and visibility masking.
- Simple U-Net head outputting `(B, K, H, W)` heatmaps.
- Masked MSE loss, PCK and pixel error evaluation.
- Per-epoch visualizations (GT vs preds) saved to disk.
- Live webcam demo with confidence-colored keypoints.
- KaggleHub downloader for the dataset.

## Setup
```bash
# install deps (consider a venv)
pip install -r requirements.txt
```

Kaggle credentials (for downloading):
- Option 1: place `kaggle.json` at `~/.kaggle/kaggle.json` (chmod 600).
- Option 2: export env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Download dataset
Using KaggleHub:
```bash
python -m src.download_dataset --dataset owner/dataset-name --out data/hand_keypoint_dataset_26k
```

Using Kaggle CLI directly:
```bash
kaggle datasets download -d owner/dataset-name -p data/hand_keypoint_dataset_26k --unzip
```

`config.yaml` should point to `images/{train,val}` and `labels/{train,val}` in that folder.

## Train
Edit `config.yaml` (num_keypoints, img_size, splits, lr, sigma, etc.). Then:
```bash
python -m src.train --config config.yaml
```
Outputs:
- Checkpoints: `outputs/checkpoints/epoch_*.pth` and `best_model.pth`
- History: `outputs/checkpoints/history.json`
- Per-epoch visuals: `outputs/checkpoints/vis/<run_tag>_epoch_<n>/sample_*.png` + epoch model.

## Evaluate
(Validation runs each epoch if `val_split` exists.) To run standalone:
```python
from torch.utils.data import DataLoader
from src.dataset.hand_kp_yolo import HandKeypointYOLODataset, collate_fn
from src.models.unet_kp import UNetKP
from src import eval as eval_utils

ds = HandKeypointYOLODataset(root="data/hand_keypoint_dataset_26k", split="val", num_keypoints=21, img_size=256)
loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
model = UNetKP(num_keypoints=21)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
metrics = eval_utils.evaluate_model(model, loader, torch.device("cpu"), pck_threshold=5.0)
print(metrics)
```

## Visualize labels/heatmaps
Overlay ground-truth heatmaps and keypoints:
```bash
python -m src.visualise --config config.yaml --idx 0      # defaults from config
python -m src.visualise --rotate-deg 30 --kp 5 --color-jitter  # extra transforms
```

## Live webcam demo
Draw predicted keypoints on webcam frames (confidence-colored red/yellow/green):
```bash
python -m src.live_demo --config config.yaml --model outputs/checkpoints/best_model.pth
# tweak confidence thresholds:
python -m src.live_demo --conf-mid 0.2 --conf-high 0.6
```

## CLI help
For full command-line options and descriptions for `visualise` and `live_demo`, see `cli_reference.md`.

## Notes
- `sigma` controls heatmap spread; larger is smoother/easier early training.
- `pck_threshold` is in pixels (matching `img_size`), adjust for use case.
- **WIP**: Works best if hand is on a plain background 

## Future Improvements
- **Data Cleaning**: Dataset has mislabelled keypoints for some photos. Need to go through with MediaPipe and mark the data points correctly
- **Robust**: Model is not robust to angle changes and different backgrounds. Will benefit form data cleaning as trickier photos are poorly labelled
