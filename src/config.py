FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# Data directories
DATA_DIR = "../data/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k"
MODEL_SAVE_PATH = f"../models/fingertip_model_"

# Image settings
IMG_SIZE = (224, 224)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training settings
BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_EPOCHS = 20