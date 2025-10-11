# 🖐 Fingertip Detection using CNNs (Work in Progress)

This project implements a **real-time fingertip detection system** using a **custom Convolutional Neural Network (CNN)** trained on the [Hand Keypoint Dataset](https://www.kaggle.com/datasets/riondsilva21/hand-keypoint-dataset-26k) to predict fingertip coordinates from RGB images. The goal is to enable **gesture-based computer interaction**, such as controlling audio or brightness levels through hand gestures.

## Current Status

**Implemented so far:**
- Built and trained a **PyTorch CNN** from scratch to regress fingertip positions from images.  
- Parsed YOLOv8 keypoint labels to extract **thumb, index, middle, ring, and little finger tips** (indices 4, 8, 12, 16, 20).  
- Added **training and validation loops** with progress tracking using `tqdm`.  
- Achieved **MSE loss ~0.01** on training/validation data, corresponding to ~13 pixel average error at 128×128 resolution.  
- Clean project structure with modular `dataset.py`, `model.py`, and `train.py`.  
- Dataset sourced and managed via **Kaggle** API.

Currently training on CPU (Qualcomm Adreno), with cloud GPU training planned for improved performance.


## How It Works

1. Images are preprocessed and normalized to 128×128 resolution.  
2. Keypoint labels are read from YOLOv8 text files, and fingertip coordinates are extracted and used as regression targets.  
3. A CNN model predicts 10 outputs — (x, y) pairs for the 5 fingertips.  
4. Model is trained with **Mean Squared Error loss** using **PyTorch**.  
5. Validation is performed after every epoch to monitor generalization and avoid overfitting.


## Tech Stack

- Python  
- PyTorch  
- NumPy, Pandas  
- OpenCV (for visualization)  
- tqdm (progress bars)  
- Kaggle Datasets


## Features in Progress / To Come

- **ResNet18 Backbone:**  
  Replace the custom CNN with a pretrained ResNet18 for better accuracy and faster convergence.

- **Live Camera Feed Integration:**  
  Use OpenCV to connect the trained model to a real-time camera feed for fingertip tracking.

- **Gesture Mapping:**  
  Map different fingertip configurations to system actions (e.g., volume control, brightness, app switching).

- **Visualization & Smoothing:**  
  Add live fingertip visualization and temporal smoothing to stabilize predictions.

- **Cloud Training:**  
  Move training to a GPU environment (Google Colab / Kaggle Notebooks) for significant speedups.


## Getting Started (Training)

1. **Install dependencies**
```bash
pip install torch torchvision torchaudio opencv-python numpy pandas tqdm
```
2. **Download dataset**
```bash
kaggle datasets download -d riondsilva21/hand-keypoint-dataset-26k
```
3. **Run training**
```bash
kaggle datasets download -d riondsilva21/hand-keypoint-dataset-26k
```
4. **Monitor loss**
- Training and validation loss are printed every epoch.
- Final model is saved as fingertip_model.pth.


## **Notes**
- Fingertip indices used: 4, 8, 12, 16, 20
- Labels are already normalized between 0–1 in YOLOv8 format.
- Current CNN achieves stable training but is limited in accuracy — moving to ResNet18 is expected to lower MSE significantly.


## **License**

This project is for **educational and research purposes**. Dataset licensed by respective sources on Kaggle.

