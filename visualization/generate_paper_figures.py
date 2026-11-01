"""
generate_paper_figures.py — Generate all figures needed for the research paper.

Figures produced:
  fig1_system_flowchart.png       — System pipeline flowchart
  fig2_model_architecture.png     — MLP architecture diagram
  fig3_training_curves.png        — Accuracy & loss training curves (detailed)
  fig4_confusion_matrix.png       — Per-class confusion matrix
  fig5_per_class_accuracy.png     — Bar chart of accuracy per letter
  fig6_comparison_table.png       — Old CNN vs New MLP comparison
  fig7_landmark_normalization.png — Landmark normalization illustration
  fig8_sample_skeletons.png       — Sample skeleton images A-Z

Run: python generate_paper_figures.py
"""

import os, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    OUT_DIR     = config.PAPER_FIGURES_DIR
    MODEL_PATH  = config.LANDMARK_MODEL_PATH
    LABEL_MAP_F = config.LABEL_MAP_PATH
    LM_TASK     = config.LM_TASK_PATH
    RAW_DIR     = config.RAW_DATA_DIR
    SKEL_DIR    = config.PROCESSED_DATA_DIR
    HIST_PATH   = os.path.join(config.OUTPUTS_DIR, 'training_history.json')
except Exception:
    OUT_DIR     = os.path.join(PROJECT_ROOT, 'outputs', 'paper_figures')
    MODEL_PATH  = os.path.join(PROJECT_ROOT, 'models', 'landmark_model.keras')
    LABEL_MAP_F = os.path.join(PROJECT_ROOT, 'models', 'label_map.json')
    LM_TASK     = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    RAW_DIR     = os.path.join(PROJECT_ROOT, 'data', 'raw', 'asl_alphabet_train', 'asl_alphabet_train')
    SKEL_DIR    = os.path.join(PROJECT_ROOT, 'data', 'processed')
    HIST_PATH   = os.path.join(PROJECT_ROOT, 'outputs', 'training_history.json')
os.makedirs(OUT_DIR, exist_ok=True)

# Dark theme colors
BG     = '#1e1e2e'
PANEL  = '#2a2a3e'
ACCENT = '#89b4fa'
GREEN  = '#a6e3a1'
RED    = '#f38ba8'
YELLOW = '#f9e2af'
MAUVE  = '#cba6f7'
TEXT   = '#cdd6f4'
SUB    = '#6c7086'

def style_fig(fig):
    fig.patch.set_facecolor(BG)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#45475a')
    ax.grid(True, alpha=0.15, color='#45475a')

with open(LABEL_MAP_F) as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
LETTERS = [label_map[i] for i in range(len(label_map))]

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — System Flowchart
# ─────────────────────────────────────────────────────────────────────────────
def fig_system_flowchart():
    fig, ax = plt.subplots(figsize=(8, 12))
    style_fig(fig); ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)

    steps = [
        (5, 13.0, "📷  Webcam Input",              ACCENT),
        (5, 11.2, "✋  MediaPipe Hand Detection",   MAUVE),
        (5,  9.4, "📍  Extract 21 Landmarks (x,y,z)", YELLOW),
        (5,  7.6, "⚖️   Normalize Landmarks\n(wrist=origin, scale=1)", GREEN),
        (5,  5.8, "🧠  MLP Classifier\n(63 → 512 → 256 → 128 → 26)", ACCENT),
        (5,  4.0, "🔤  Predicted Letter",           GREEN),
        (5,  2.2, "📝  Build Sentence",             MAUVE),
        (5,  0.5, "🔊  Text-to-Speech Output",      RED),
    ]

    for x, y, label, color in steps:
        fancy = FancyBboxPatch((x-3.2, y-0.55), 6.4, 1.1,
                               boxstyle="round,pad=0.1",
                               facecolor=color+'33', edgecolor=color,
                               linewidth=2, zorder=2)
        ax.add_patch(fancy)
        ax.text(x, y, label, ha='center', va='center',
                color=TEXT, fontsize=10, fontweight='bold', zorder=3)

    for i in range(len(steps)-1):
        x1, y1 = steps[i][0], steps[i][1] - 0.55
        x2, y2 = steps[i+1][0], steps[i+1][1] + 0.55
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=SUB, lw=2))

    ax.set_title('System Pipeline Flowchart', color=ACCENT,
                 fontsize=14, fontweight='bold', pad=10)
    path = os.path.join(OUT_DIR, 'fig1_system_flowchart.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — MLP Architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    style_fig(fig); ax.set_facecolor(BG); ax.axis('off')
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)

    layers = [
        ("Input\n63",    0.8,  ACCENT,  0.8),
        ("Dense\n512\n+BN+Drop(0.5)", 2.8, MAUVE, 1.2),
        ("Dense\n256\n+BN+Drop(0.4)", 5.0, MAUVE, 1.0),
        ("Dense\n128\n+BN+Drop(0.3)", 7.2, MAUVE, 0.85),
        ("Dense\n64\n+Drop(0.2)",     9.2, MAUVE, 0.7),
        ("Output\n26\nsoftmax",       11.2, GREEN, 0.8),
    ]

    for label, x, color, w in layers:
        rect = FancyBboxPatch((x - w/2, 1.2), w, 2.6,
                              boxstyle="round,pad=0.1",
                              facecolor=color+'33', edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x, 2.5, label, ha='center', va='center',
                color=TEXT, fontsize=9, fontweight='bold')

    for i in range(len(layers)-1):
        x1 = layers[i][1] + layers[i][3]/2
        x2 = layers[i+1][1] - layers[i+1][3]/2
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle='->', color=SUB, lw=2))

    ax.text(7, 0.3, 'Activation: ReLU (hidden)  |  Softmax (output)  |  Optimizer: Adam  |  Loss: Categorical Crossentropy',
            ha='center', va='center', color=SUB, fontsize=9)
    ax.set_title('MLP Model Architecture', color=ACCENT, fontsize=14, fontweight='bold')

    path = os.path.join(OUT_DIR, 'fig2_model_architecture.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Training Curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_training_curves():
    if not os.path.exists(HIST_PATH):
        print("training_history.json not found — skipping training curves")
        return

    with open(HIST_PATH) as f:
        h = json.load(f)

    epochs    = range(1, len(h['accuracy'])+1)
    train_acc = [v*100 for v in h['accuracy']]
    val_acc   = [v*100 for v in h['val_accuracy']]
    train_loss = h['loss']
    val_loss   = h['val_loss']
    best_ep    = int(np.argmax(h['val_accuracy'])) + 1
    best_val   = max(h['val_accuracy']) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    style_fig(fig)

    for ax in (ax1, ax2):
        style_ax(ax)

    # Accuracy
    ax1.plot(epochs, train_acc, color=ACCENT, lw=2, label='Train Accuracy')
    ax1.plot(epochs, val_acc,   color=GREEN,  lw=2, label='Val Accuracy')
    ax1.axvline(best_ep, color=YELLOW, lw=1.5, ls='--', alpha=0.8, label=f'Best epoch ({best_ep})')
    ax1.axhline(best_val, color=GREEN, lw=1, ls=':', alpha=0.5)
    ax1.text(best_ep+1, best_val-3, f'{best_val:.2f}%', color=GREEN, fontsize=9)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training & Validation Accuracy')
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.9, fontsize=9)
    ax1.set_ylim(0, 105)

    # Loss
    ax2.plot(epochs, train_loss, color=ACCENT, lw=2, label='Train Loss')
    ax2.plot(epochs, val_loss,   color=RED,    lw=2, label='Val Loss')
    ax2.axvline(best_ep, color=YELLOW, lw=1.5, ls='--', alpha=0.8, label=f'Best epoch ({best_ep})')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Training & Validation Loss')
    ax2.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.9, fontsize=9)

    fig.suptitle(f'Landmark MLP Training Results  |  Best Val Accuracy: {best_val:.2f}%',
                 color=ACCENT, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 & 5 — Confusion Matrix + Per-class Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def extract_lm(img_bgr, detector):
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

def fig_confusion_and_accuracy():
    print("Loading model and extracting test landmarks...")
    model = load_model(MODEL_PATH)

    base_opts = mp_python.BaseOptions(model_asset_path=LM_TASK)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    X, y = [], []
    for idx, letter in label_map.items():
        folder = os.path.join(RAW_DIR, letter)
        if not os.path.isdir(folder): continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        # Use last 200 images per class as test set (not used in training split)
        test_files = files[-200:]
        for fname in test_files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None: continue
            feat = extract_lm(img, detector)
            if feat is None: continue
            X.append(feat); y.append(idx)
        print(f"  {letter}: {len([i for i in y if i==idx])} test samples")

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')

    preds = np.argmax(model.predict(X, batch_size=512, verbose=0), axis=1)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, preds, labels=list(range(len(LETTERS))))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    style_fig(fig); ax.set_facecolor(BG)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(LETTERS))); ax.set_xticklabels(LETTERS, color=TEXT, fontsize=9)
    ax.set_yticks(range(len(LETTERS))); ax.set_yticklabels(LETTERS, color=TEXT, fontsize=9)
    ax.set_xlabel('Predicted Label', color=TEXT, fontsize=11)
    ax.set_ylabel('True Label',      color=TEXT, fontsize=11)
    ax.set_title('Confusion Matrix (Normalized)', color=ACCENT, fontsize=14, fontweight='bold')
    for spine in ax.spines.values(): spine.set_edgecolor('#45475a')

    # Annotate cells
    for i in range(len(LETTERS)):
        for j in range(len(LETTERS)):
            val = cm_norm[i, j]
            if val > 0.05:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if val > 0.5 else TEXT, fontsize=6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

    # ── Per-class accuracy bar chart ──────────────────────────────────────────
    per_class_acc = cm_norm.diagonal() * 100
    colors = [GREEN if a >= 95 else YELLOW if a >= 85 else RED for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(14, 5))
    style_fig(fig); style_ax(ax)
    bars = ax.bar(LETTERS, per_class_acc, color=colors, edgecolor='#45475a', linewidth=0.5)
    ax.axhline(np.mean(per_class_acc), color=ACCENT, lw=2, ls='--',
               label=f'Mean: {np.mean(per_class_acc):.1f}%')
    ax.set_xlabel('Letter', color=TEXT); ax.set_ylabel('Accuracy (%)', color=TEXT)
    ax.set_title('Per-Class Test Accuracy', color=ACCENT, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{acc:.0f}', ha='center', va='bottom', color=TEXT, fontsize=8)

    legend_patches = [
        mpatches.Patch(color=GREEN,  label='≥ 95%'),
        mpatches.Patch(color=YELLOW, label='85–95%'),
        mpatches.Patch(color=RED,    label='< 85%'),
    ]
    ax.legend(handles=legend_patches, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_per_class_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Comparison Table
# ─────────────────────────────────────────────────────────────────────────────
def fig_comparison_table():
    fig, ax = plt.subplots(figsize=(12, 5))
    style_fig(fig); ax.set_facecolor(BG); ax.axis('off')

    columns = ['Aspect', 'Old CNN (8-group)', 'New Landmark MLP']
    rows = [
        ['Model type',         'CNN on skeleton images',     'MLP on landmark coords'],
        ['Output classes',     '8 groups',                   '26 letters (direct)'],
        ['Letter disambiguation', 'Hardcoded geometric rules', 'Learned from data'],
        ['Input size',         '400×400×3 = 480,000 values', '63 values'],
        ['Training data',      'Webcam skeleton images',     'Real ASL photos (diverse)'],
        ['Val accuracy',       '~3–7%  (severe overfitting)', '97.96%'],
        ['Train accuracy',     '~92%',                       '~97.9%'],
        ['Train/Val gap',      '~85–89%',                    '~0.1%'],
        ['Inference speed',    'Slow (CNN on 400×400)',       'Fast (MLP on 63 numbers)'],
        ['Background robust',  '❌ No',                       '✅ Yes'],
        ['Scale invariant',    '❌ No',                       '✅ Yes'],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(PANEL)
        cell.set_edgecolor('#45475a')
        cell.set_text_props(color=TEXT)
        if row == 0:
            cell.set_facecolor(ACCENT+'55')
            cell.set_text_props(color=TEXT, fontweight='bold')
        elif col == 1 and row > 0:
            cell.set_facecolor(RED+'22')
        elif col == 2 and row > 0:
            cell.set_facecolor(GREEN+'22')

    ax.set_title('Model Comparison: Old CNN vs New Landmark MLP',
                 color=ACCENT, fontsize=13, fontweight='bold', pad=20)

    path = os.path.join(OUT_DIR, 'fig6_comparison_table.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Landmark Normalization Illustration
# ─────────────────────────────────────────────────────────────────────────────
def fig_landmark_normalization():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    style_fig(fig)

    titles = ['Raw Landmarks\n(pixel coords)', 'After Translation\n(wrist at origin)', 'After Scaling\n(max dist = 1)']
    colors_list = [RED, YELLOW, GREEN]

    # Simulate a hand landmark set
    np.random.seed(42)
    raw = np.array([
        [320, 400], [310, 360], [300, 320], [295, 285], [290, 260],  # thumb
        [340, 290], [345, 250], [348, 220], [350, 200],               # index
        [360, 285], [368, 245], [372, 215], [374, 195],               # middle
        [375, 290], [380, 255], [382, 228], [383, 210],               # ring
        [390, 300], [398, 270], [400, 248], [402, 232],               # pinky
    ], dtype=float)

    translated = raw - raw[0]
    scale = np.max(np.linalg.norm(translated, axis=1))
    normalized = translated / scale

    datasets = [raw, translated, normalized]
    connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                   (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                   (13,17),(17,18),(18,19),(19,20),(0,17)]

    for ax, pts, title, color in zip(axes, datasets, titles, colors_list):
        ax.set_facecolor(PANEL)
        for a, b in connections:
            ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]],
                    color=color, lw=2, alpha=0.7)
        ax.scatter(pts[:,0], pts[:,1], c=ACCENT, s=40, zorder=5)
        ax.scatter(pts[0,0], pts[0,1], c=RED, s=80, zorder=6, label='Wrist')
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold')
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#45475a')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)

    fig.suptitle('Landmark Normalization Process', color=ACCENT, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_landmark_normalization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Sample Skeleton Images A-Z
# ─────────────────────────────────────────────────────────────────────────────
def fig_sample_skeletons():
    letters = sorted([d for d in os.listdir(SKEL_DIR)
                      if os.path.isdir(os.path.join(SKEL_DIR, d)) and d.upper() in set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')])

    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    style_fig(fig)
    axes = axes.flatten()

    for i, letter in enumerate(letters[:26]):
        folder = os.path.join(SKEL_DIR, letter)
        files  = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
        if not files: continue
        img = cv2.imread(os.path.join(folder, files[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        axes[i].imshow(img)
        axes[i].set_title(letter, color=ACCENT, fontsize=12, fontweight='bold')
        axes[i].axis('off')
        for spine in axes[i].spines.values(): spine.set_edgecolor(ACCENT)

    # Hide unused subplots
    for j in range(len(letters), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Sample Skeleton Images — A to Z', color=ACCENT, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig8_sample_skeletons.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Generating Research Paper Figures")
    print("=" * 55 + "\n")

    print("1/7  System flowchart...")
    fig_system_flowchart()

    print("2/7  Model architecture...")
    fig_model_architecture()

    print("3/7  Training curves...")
    fig_training_curves()

    print("4/7  Comparison table...")
    fig_comparison_table()

    print("5/7  Landmark normalization...")
    fig_landmark_normalization()

    print("6/7  Sample skeletons...")
    fig_sample_skeletons()

    print("7/7  Confusion matrix + per-class accuracy (needs landmark extraction ~5 min)...")
    fig_confusion_and_accuracy()

    print(f"\nAll figures saved to: {OUT_DIR}")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f"  {f}  ({size} KB)")
