"""
test_accuracy.py — Tests the landmark MLP model accuracy.
Run: python test_accuracy.py
"""

import os, json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    MODEL_PATH  = config.LANDMARK_MODEL_PATH
    LABEL_MAP_F = config.LABEL_MAP_PATH
    LM_TASK     = config.LM_TASK_PATH
    RAW_DIR     = config.RAW_DATA_DIR
except Exception:
    MODEL_PATH  = os.path.join(PROJECT_ROOT, 'models', 'landmark_model.keras')
    LABEL_MAP_F = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
    LM_TASK     = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    RAW_DIR     = os.path.join(PROJECT_ROOT, 'data', 'raw', 'asl_alphabet_train', 'asl_alphabet_train')

with open(LABEL_MAP_F) as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
LETTERS = [label_map[i] for i in range(len(label_map))]

# MediaPipe detector
base_opts = mp_python.BaseOptions(model_asset_path=LM_TASK)
opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts, num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(opts)

def extract_lm(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None
    lms = result.hand_landmarks[0]
    pts = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in lms], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6: return None
    pts /= scale
    return pts.flatten()

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.\n")
print("Testing on last 200 images per letter (unseen test set)...\n")

letter_results = {}
total_correct = total_images = 0

for idx, letter in label_map.items():
    folder = os.path.join(RAW_DIR, letter)
    if not os.path.isdir(folder): continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    test_files = files[-200:]  # last 200 = held-out test set

    X, y = [], []
    for fname in test_files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None: continue
        feat = extract_lm(img)
        if feat is None: continue
        X.append(feat); y.append(idx)

    if not X: continue
    X = np.array(X, dtype='float32')
    preds = np.argmax(model.predict(X, batch_size=256, verbose=0), axis=1)
    correct = int(np.sum(preds == idx))
    total   = len(X)
    acc     = correct / total * 100

    letter_results[letter] = (correct, total, acc)
    total_correct += correct
    total_images  += total

    bar = '█' * int(acc/5) + '░' * (20 - int(acc/5))
    print(f"  {letter}  [{bar}]  {acc:5.1f}%  ({correct}/{total})")

# Summary
overall = total_correct / total_images * 100 if total_images else 0
print("\n" + "="*55)
print(f"  OVERALL ACCURACY : {overall:.2f}%")
print(f"  Total correct    : {total_correct}/{total_images}")
print("="*55)

# Weak letters
weak = [(l, r) for l, r in letter_results.items() if r[2] < 90]
print("\n  Letters below 90%:")
if weak:
    for l, (c, t, a) in sorted(weak, key=lambda x: x[1][2]):
        print(f"    {l}: {a:.1f}%  ({c}/{t})")
else:
    print("    None — all letters above 90%! ✓")
