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

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    MODEL_PATH  = config.CNN_MODEL_PATH
    BACKUP_PATH = config.CNN_BACKUP_PATH
    DATA_DIR    = config.PROCESSED_DATA_DIR
    PLOT_PATH   = os.path.join(config.OUTPUTS_DIR, 'training_results.png')
except Exception:
    MODEL_PATH  = os.path.join(PROJECT_ROOT, 'models', 'cnn8grps_rad1_model.h5')
    BACKUP_PATH = os.path.join(PROJECT_ROOT, 'models', 'cnn8grps_rad1_model_backup.h5')
    DATA_DIR    = os.path.join(PROJECT_ROOT, 'data', 'processed')
    PLOT_PATH   = os.path.join(PROJECT_ROOT, 'outputs', 'training_results.png')

IMG_SIZE   = 400
BATCH_SIZE = 8      # small batch = low RAM usage
EPOCHS     = 40

LETTER_TO_GROUP = {
    'A':0,'E':0,'M':0,'N':0,'S':0,'T':0,
    'B':1,'D':1,'F':1,'I':1,'K':1,'R':1,'U':1,'V':1,'W':1,
    'C':2,'O':2, 'G':3,'H':3, 'L':4,
    'P':5,'Q':5,'Z':5, 'X':6, 'Y':7,'J':7,
}

# ── Collect all file paths + labels (no images loaded yet) ───────────────────
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

# ── Batch generator — loads & augments on the fly ────────────────────────────
class BatchGen:
    def __init__(self, paths, labels, batch_size, augment=False):
        self.paths      = paths
        self.labels     = labels
        self.batch_size = batch_size
        self.augment    = augment
        self.aug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.10,
            height_shift_range=0.10,
            zoom_range=0.15,
            brightness_range=[0.6, 1.4],
            shear_range=5,
            fill_mode='nearest',
        ) if augment else None
        self.indices = np.arange(len(paths))
        np.random.shuffle(self.indices)
        self.pos = 0

    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= len(self.indices):
            self.pos = 0
            np.random.shuffle(self.indices)
        idx   = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        imgs, labs = [], []
        for i in idx:
            img = cv2.imread(self.paths[i])
            if img is None: continue
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            if self.augment and self.aug:
                img = self.aug.random_transform(img)
            imgs.append(img)
            labs.append(self.labels[i])
        return np.array(imgs), to_categorical(labs, 8)

def train():
    print("="*55)
    print("  Sign Language CNN — Fine-tuning")
    print("="*55)

    # Backup
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"\nBackup saved → {BACKUP_PATH}")

    # Collect paths
    print("\nScanning dataset...")
    paths, labels = collect_paths()
    print(f"Found {len(paths)} images across {len(set(labels))} groups")

    # Split
    p_train, p_val, l_train, l_val = train_test_split(
        paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"Train: {len(p_train)}  |  Val: {len(p_val)}")

    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Generators
    train_gen = BatchGen(p_train, l_train, BATCH_SIZE, augment=True)
    val_gen   = BatchGen(p_val,   l_val,   BATCH_SIZE, augment=False)

    # Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs...\n")

    # Manual training loop (works with our generator)
    best_val_acc = 0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # Train
        t_acc, t_loss, steps = 0, 0, 0
        for _ in range(len(train_gen)):
            X, y = next(train_gen)
            if len(X) == 0: continue
            result = model.train_on_batch(X, y)
            t_loss += result[0]; t_acc += result[1]; steps += 1
        t_loss /= max(steps,1); t_acc /= max(steps,1)

        # Validate
        v_acc, v_loss, vsteps = 0, 0, 0
        for _ in range(len(val_gen)):
            X, y = next(val_gen)
            if len(X) == 0: continue
            result = model.test_on_batch(X, y)
            v_loss += result[0]; v_acc += result[1]; vsteps += 1
        v_loss /= max(vsteps,1); v_acc /= max(vsteps,1)

        train_accs.append(t_acc);   val_accs.append(v_acc)
        train_losses.append(t_loss); val_losses.append(v_loss)

        print(f"  loss={t_loss:.4f}  acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

        # Save best
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            model.save(MODEL_PATH)
            print(f"  ✓ Saved best model (val_acc={v_acc*100:.2f}%)")
            no_improve = 0
        else:
            no_improve = getattr(train, '_no_improve', 0) + 1
            train._no_improve = no_improve
            if no_improve >= 10:
                print("  Early stopping.")
                break

    print(f"\n{'='*55}")
    print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"  Model saved → {MODEL_PATH}")
    print(f"{'='*55}\n")

    # Plot
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
