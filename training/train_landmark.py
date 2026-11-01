"""
train_landmark.py — Train a 26-class MLP on MediaPipe hand landmarks.

Approach:
  1. Extract 21 (x,y,z) landmarks from real hand photos (archive dataset)
  2. Normalize: translate wrist to origin, scale by max distance
  3. Train a small MLP — 63 features → 26 classes
  4. No image memorization possible → no overfitting

This is the correct approach. The CNN on skeleton images overfits because
all skeleton images were captured in one session (same background/lighting).
Landmark coordinates are invariant to background, lighting, and hand size.

Run:
    python train_landmark.py

Output:
    landmark_model.keras   — saved model
    label_map.json         — {0: 'A', 1: 'B', ...}
    landmark_training.png  — accuracy/loss curves
"""

import os, json, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    RAW_DIR     = config.RAW_DATA_DIR
    MODEL_OUT   = config.LANDMARK_MODEL_PATH
    LABEL_MAP_F = config.LABEL_MAP_PATH
    PLOT_OUT    = os.path.join(config.OUTPUTS_DIR, 'landmark_training.png')
    LM_TASK     = config.LM_TASK_PATH
except Exception:
    RAW_DIR     = os.path.join(PROJECT_ROOT, 'data', 'raw', 'asl_alphabet_train', 'asl_alphabet_train')
    MODEL_OUT   = os.path.join(PROJECT_ROOT, 'models', 'landmark_model.keras')
    LABEL_MAP_F = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
    PLOT_OUT    = os.path.join(PROJECT_ROOT, 'outputs', 'landmark_training.png')
    LM_TASK     = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')

VALID_LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NUM_FEATURES  = 63   # 21 landmarks × (x, y, z)
BATCH_SIZE    = 512
EPOCHS        = 200

# ── MediaPipe landmarker (image mode, low confidence for cropped images) ──────
base_options = mp_python.BaseOptions(model_asset_path=LM_TASK)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    min_tracking_confidence=0.1,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(img_bgr):
    """
    Returns normalized 63-d feature vector or None.
    Normalization: wrist at origin, max landmark distance = 1.
    This makes the vector invariant to hand size, position, and distance.
    """
    h, w = img_bgr.shape[:2]
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float32)

    # Translate wrist (landmark 0) to origin
    pts -= pts[0]

    # Scale so max distance from wrist = 1
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        return None
    pts /= scale

    return pts.flatten()


def load_dataset():
    print(f"Extracting landmarks from: {RAW_DIR}")
    print("(This runs once — ~10-20 min for 87k images)\n")

    letters = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d)) and d.upper() in VALID_LETTERS
    ])
    label_map     = {i: l for i, l in enumerate(letters)}
    label_map_inv = {l: i for i, l in label_map.items()}

    print(f"Classes ({len(letters)}): {letters}\n")

    X, y = [], []
    t0 = time.time()

    for letter in letters:
        folder = os.path.join(RAW_DIR, letter)
        idx    = label_map_inv[letter]
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        ok = fail = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                fail += 1
                continue
            feat = extract_landmarks(img)
            if feat is None:
                fail += 1
                continue
            X.append(feat)
            y.append(idx)
            ok += 1
        pct = ok / len(files) * 100 if files else 0
        print(f"  {letter}: {ok}/{len(files)} ({pct:.0f}%)  [{fail} no hand]")

    elapsed = time.time() - t0
    print(f"\nExtracted {len(X)} samples in {elapsed/60:.1f} min")
    return np.array(X, dtype='float32'), np.array(y, dtype='int32'), label_map


def build_model(num_classes):
    """
    Small MLP — 63 inputs is tiny, no need for a CNN.
    Regularized heavily to prevent overfitting.
    """
    model = Sequential([
        Input(shape=(NUM_FEATURES,)),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train():
    print("=" * 55)
    print("  Sign Language — Landmark MLP (26 classes)")
    print("=" * 55 + "\n")

    X, y, label_map = load_dataset()
    num_classes = len(label_map)

    # Save label map
    with open(LABEL_MAP_F, 'w') as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"\nLabel map saved → {LABEL_MAP_F}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    Y_tr  = to_categorical(y_tr,  num_classes)
    Y_val = to_categorical(y_val, num_classes)
    print(f"Train: {len(X_tr)}  |  Val: {len(X_val)}\n")

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    print(f"\nTraining MLP on {len(X_tr)} landmark vectors...\n")
    t0 = time.time()

    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed/60:.1f} min")

    # Save history for plotting
    hist_path = os.path.join(BASE_DIR, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"History saved → {hist_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=120)
    print(f"Plot saved → {PLOT_OUT}")

    best_val = max(history.history['val_accuracy'])
    print(f"\n{'=' * 55}")
    print(f"  Best val accuracy : {best_val * 100:.2f}%")
    print(f"  Model saved       → {MODEL_OUT}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    train()
