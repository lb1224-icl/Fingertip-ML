# CLI Reference

Command-line options for the visualization and live demo tools.

## Visualize (`src/visualise.py`)
```
python -m src.visualise [options]
```
- `--config <path>`: Path to config file (`data/config.yaml` by default).
- `--root <path>`: Dataset root (overrides config).
- `--split <name>`: Dataset split, e.g., `train` or `val` (overrides config).
- `--idx <int>`: Sample index to visualize (default: 0).
- `--kp <int>`: Keypoint index to show; `-1` shows max over all keypoints (default: -1).
- `--img-size <int>`: Resize side for images/heatmaps; falls back to config (default: None).
- `--sigma <float>`: Gaussian sigma for heatmaps; falls back to config (default: None).
- `--num-keypoints <int>`: Number of keypoints; falls back to config (default: None).
- `--save <path>`: Optional path to save the figure (default: None).
- `--color-jitter`: Enable color jitter augmentation for the sample (default: False).
- `--augment`: Use dataset augmentations (random hflip) for the sample (default: False).
- `--rotate-deg <float>`: Rotate sample by degrees CCW and regenerate heatmaps (default: 0.0).

## Live Demo (`src/live_demo.py`)
```
python -m src.live_demo [options]
```
- `--model <path>`: Path to model checkpoint (`outputs/checkpoints/best_model.pth` by default).
- `--config <path>`: Path to config file (`data/config.yaml` by default).
- `--cam <int>`: Camera index (0 is default).
- `--img-size <int>`: Resize side for model input; falls back to config (default: None).
- `--num-keypoints <int>`: Override number of keypoints; falls back to config (default: None).
- `--conf-mid <float>`: Confidence threshold for yellow markers (default: 0.2).
- `--conf-high <float>`: Confidence threshold for green markers (default: 0.5).
