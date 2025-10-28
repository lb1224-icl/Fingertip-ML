
import os

# ===== DATA PATHS =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hand_keypoint_dataset_26k", "hand_keypoint_dataset_26k"))
TRAIN_IMAGES = os.path.join(BASE_DIR, "images", "train")
VAL_IMAGES = os.path.join(BASE_DIR, "images", "val")
TRAIN_LABELS = os.path.join(BASE_DIR, "labels", "train")
VAL_LABELS = os.path.join(BASE_DIR, "labels", "val")

# ===== MODEL & TRAINING =====
SAVE_MODEL_PATH = os.path.join(BASE_DIR, "models", "fingertip_best.pth")
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50

# ===== IMAGE SETTINGS =====
IMG_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ===== FINGERTIP SETTINGS =====
# order: thumb, index, middle, ring, pinky
FINGERTIP_INDICES = [4, 8, 12, 16, 20]   # original dataset indices (for mapping only)
NUM_FINGERTIPS = 5
NUM_OUTPUTS = NUM_FINGERTIPS * 3  # x,y,v per fingertip

# ===== VISIBILITY CHECK SETTINGS =====
PATCH_RADIUS = 5
BRIGHTNESS_THRESHOLD = 8.0


