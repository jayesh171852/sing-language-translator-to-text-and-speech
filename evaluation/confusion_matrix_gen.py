"""
confusion_matrix_gen.py — Generate confusion matrix for the landmark MLP model.
Run: python confusion_matrix_gen.py
"""

import os, json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    OUT_DIR     = config.PAPER_FIGURES_DIR
except Exception:
    MODEL_PATH  = os.path.join(PROJECT_ROOT, 'models', 'landmark_model.keras')
    LABEL_MAP_F = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
    LM_TASK     = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    RAW_DIR     = os.path.join(PROJECT_ROOT, 'data', 'raw', 'asl_alphabet_train', 'asl_alphabet_train')
    OUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

BG    = '#1e1e2e'
PANEL = '#2a2a3e'
TEXT  = '#cdd6f4'
ACCENT= '#89b4fa'
GREEN = '#a6e3a1'
RED   = '#f38ba8'

with open(LABEL_MAP_F) as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
LETTERS = [label_map[i] for i in range(len(label_map))]

# ── MediaPipe ─────────────────────────────────────────────────────────────────
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

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.\n")
print("Extracting test landmarks (last 200 images per letter)...")

X_test, y_test = [], []

for idx, letter in label_map.items():
    folder = os.path.join(RAW_DIR, letter)
    if not os.path.isdir(folder): continue
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])
    test_files = files[-200:]
    ok = 0
    for fname in test_files:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None: continue
        feat = extract_lm(img)
        if feat is None: continue
        X_test.append(feat)
        y_test.append(idx)
        ok += 1
    print(f"  {letter}: {ok} samples")

X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test, dtype='int32')

print(f"\nTotal test samples: {len(X_test)}")
print("Running predictions...")

y_pred = np.argmax(model.predict(X_test, batch_size=512, verbose=0), axis=1)

# ── Overall accuracy ──────────────────────────────────────────────────────────
overall = np.mean(y_pred == y_test) * 100
print(f"\nOverall Test Accuracy: {overall:.2f}%")

# Per-letter accuracy
print("\nPer-letter accuracy:")
print("-" * 35)
for idx, letter in label_map.items():
    mask = y_test == idx
    if mask.sum() == 0: continue
    acc = np.mean(y_pred[mask] == idx) * 100
    print(f"  {letter}: {acc:.1f}%  ({int(mask.sum() * acc/100)}/{mask.sum()})")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm      = confusion_matrix(y_test, y_pred, labels=list(range(len(LETTERS))))
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Plot 1: Normalized confusion matrix
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors=TEXT)
cbar.set_label('Proportion', color=TEXT, fontsize=10)

ax.set_xticks(range(len(LETTERS)))
ax.set_yticks(range(len(LETTERS)))
ax.set_xticklabels(LETTERS, color=TEXT, fontsize=11, fontweight='bold')
ax.set_yticklabels(LETTERS, color=TEXT, fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted Label', color=TEXT, fontsize=13, labelpad=10)
ax.set_ylabel('True Label',      color=TEXT, fontsize=13, labelpad=10)
ax.set_title(f'Confusion Matrix — Landmark MLP  |  Overall Accuracy: {overall:.2f}%',
             color=ACCENT, fontsize=14, fontweight='bold', pad=15)

for spine in ax.spines.values():
    spine.set_edgecolor('#45475a')

# Annotate cells with values
for i in range(len(LETTERS)):
    for j in range(len(LETTERS)):
        val = cm_norm[i, j]
        if val > 0.02:
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if val > 0.5 else TEXT,
                    fontweight='bold' if i == j else 'normal')

plt.tight_layout()
path1 = os.path.join(OUT_DIR, 'confusion_matrix_normalized.png')
plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"\nSaved: {path1}")

# Plot 2: Per-class accuracy bar chart
per_class_acc = cm_norm.diagonal() * 100
colors = [GREEN if a >= 95 else '#f9e2af' if a >= 85 else RED for a in per_class_acc]

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

bars = ax.bar(LETTERS, per_class_acc, color=colors, edgecolor='#45475a', linewidth=0.6)
ax.axhline(overall, color=ACCENT, lw=2, ls='--', label=f'Overall: {overall:.2f}%')
ax.axhline(95, color=GREEN, lw=1, ls=':', alpha=0.5)

for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc:.0f}', ha='center', va='bottom', color=TEXT, fontsize=8, fontweight='bold')

ax.set_xlabel('Letter', color=TEXT, fontsize=12)
ax.set_ylabel('Accuracy (%)', color=TEXT, fontsize=12)
ax.set_title('Per-Letter Test Accuracy', color=ACCENT, fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values(): spine.set_edgecolor('#45475a')
ax.grid(True, axis='y', alpha=0.15, color='#45475a')

import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=GREEN,    label='>= 95%'),
    mpatches.Patch(color='#f9e2af',label='85-95%'),
    mpatches.Patch(color=RED,      label='< 85%'),
    mpatches.Patch(color=ACCENT,   label=f'Overall: {overall:.2f}%'),
]
ax.legend(handles=legend_patches, facecolor=PANEL, labelcolor=TEXT,
          framealpha=0.9, fontsize=9, loc='lower right')

plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'per_letter_accuracy.png')
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path2}")

print(f"\nDone! Overall accuracy: {overall:.2f}%")
