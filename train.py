"""
train.py — Fine-tune the sign language CNN using a batch generator.
Loads images on-demand so it never runs out of RAM.
Run: python train.py
"""
import os, shutil, math
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')
BACKUP_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model_backup.h5')
DATA_DIR    = os.path.join(BASE_DIR, 'AtoZ_3.1')
PLOT_PATH   = os.path.join(BASE_DIR, 'training_results.png')

IMG_SIZE   = 400
BATCH_SIZE = 64     # larger batch = much faster per epoch
EPOCHS     = 200
TARGET_ACC = 0.90         # stop when val accuracy hits 90%

LETTER_TO_GROUP = {
    'A':0,'E':0,'M':0,'N':0,'S':0,'T':0,
    'B':1,'D':1,'F':1,'I':1,'K':1,'R':1,'U':1,'V':1,'W':1,
    'C':2,'O':2, 'G':3,'H':3, 'L':4,
    'P':5,'Q':5,'Z':5, 'X':6, 'Y':7,'J':7,
}

def collect_paths():
    paths, labels = [], []
    for letter in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, letter)
        if not os.path.isdir(d): continue
        g = LETTER_TO_GROUP.get(letter.upper())
        if g is None: continue
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                paths.append(os.path.join(d, f))
                labels.append(g)
    return np.array(paths), np.array(labels)

aug_gen = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.10,
    height_shift_range=0.10, zoom_range=0.15,
    brightness_range=[0.6, 1.4], shear_range=5,
    fill_mode='nearest',
)

def load_batch(paths, labels, augment=False):
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        if augment:
            img = aug_gen.random_transform(img)
        imgs.append(img)
    return np.array(imgs), to_categorical(labels[:len(imgs)], 8)

def train():
    print("="*55)
    print("  Sign Language CNN — Fine-tuning")
    print("="*55)

    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Backup saved → {BACKUP_PATH}")

    print("\nScanning dataset...")
    paths, labels = collect_paths()
    print(f"Found {len(paths)} images")

    p_train, p_val, l_train, l_val = train_test_split(
        paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"Train: {len(p_train)}  |  Val: {len(p_val)}")

    print("\nBuilding fresh model...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
    model = Sequential([
        Input(shape=(400, 400, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    best_val_acc = 0
    no_improve   = 0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"\nTraining for up to {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        idx = np.random.permutation(len(p_train))
        t_acc, t_loss, steps = 0.0, 0.0, 0
        for i in range(0, len(idx), BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            X, y = load_batch(p_train[batch_idx], l_train[batch_idx], augment=True)
            if len(X) == 0: continue
            r = model.train_on_batch(X, y)
            t_loss += r[0]; t_acc += r[1]; steps += 1
        t_loss /= max(steps,1); t_acc /= max(steps,1)

        # ── Validate ───────────────────────────────────────────────────────
        vidx = np.arange(len(p_val))
        v_acc, v_loss, vsteps = 0.0, 0.0, 0
        for i in range(0, len(vidx), BATCH_SIZE):
            batch_idx = vidx[i:i+BATCH_SIZE]
            X, y = load_batch(p_val[batch_idx], l_val[batch_idx], augment=False)
            if len(X) == 0: continue
            r = model.test_on_batch(X, y)
            v_loss += r[0]; v_acc += r[1]; vsteps += 1
        v_loss /= max(vsteps,1); v_acc /= max(vsteps,1)

        train_accs.append(t_acc);    val_accs.append(v_acc)
        train_losses.append(t_loss); val_losses.append(v_loss)

        print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
              f"loss={t_loss:.4f}  acc={t_acc*100:.1f}%  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc*100:.1f}%  "
              f"{'↑ BEST' if v_acc > best_val_acc else ''}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            model.save(MODEL_PATH)
            print(f"  ✓ Best model saved  (val_acc={v_acc*100:.2f}%)")
            no_improve = 0
            if v_acc >= TARGET_ACC:
                print(f"\n  🎯 Reached {TARGET_ACC*100:.0f}% target! Stopping.")
                break
        else:
            no_improve += 1
            if no_improve >= 15:
                print("  Early stopping — no improvement for 15 epochs.")
                break

    print(f"\n{'='*55}")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Model saved       → {MODEL_PATH}")
    print(f"{'='*55}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_accs, label='Train'); ax1.plot(val_accs, label='Val')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(train_losses, label='Train'); ax2.plot(val_losses, label='Val')
    ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Plot saved → {PLOT_PATH}")

if __name__ == '__main__':
    train()
