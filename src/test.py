import cv2
import torch
import numpy as np
from torchvision import transforms

from model import FingertipResNet
import config

CAMERA_INDEX = 0       # Default webcam
CONFIDENCE_RADIUS = 6  
DISPLAY_SCALE = 1.0   

model = "fingertip_model_" # model path name

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

model = FingertipResNet(num_outputs=10, pretrained=False).to(device)

MODEL_PATH = f"{config.MODEL_SAVE_PATH}/CURRENT_MODEL.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
])

def denormalize_for_display(img_tensor):
    img = img_tensor.clone().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    img = (img[:, :, ::-1] * 255).astype(np.uint8)
    return np.ascontiguousarray(img)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame capture failed.")
        break

    if DISPLAY_SCALE != 1.0:
        frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    h, w, _ = frame.shape

    # Preprocess for model 
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    # Predict fingertip positions 
    with torch.no_grad():
        preds = model(img_pil).cpu().numpy().reshape(-1, 2)

    # Overlay predictions 
    for i, (x, y) in enumerate(preds):
        px = int(x * w)
        py = int(y * h)
        cv2.circle(frame, (px, py), CONFIDENCE_RADIUS, (0, 0, 255), -1)
        cv2.putText(frame, f"{i}", (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Fingertip Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()