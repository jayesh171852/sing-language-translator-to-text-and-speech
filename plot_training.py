"""
plot_training.py — Regenerate a detailed training graph from saved history.
Run: python plot_training.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
HIST_PATH  = os.path.join(BASE_DIR, 'training_history.json')
MODEL_PATH = os.path.join(BASE_DIR, 'landmark_model.keras')
OUT_PATH   = os.path.join(BASE_DIR, 'landmark_training_detailed.png')

# ── Load history ──────────────────────────────────────────────────────────────
if not os.path.exists(HIST_PATH):
    print(f"History file not found at {HIST_PATH}")
    print("Generating from existing landmark_training.png instead...")
    # We'll re-run a short training to capture history
    import sys
    sys.exit(1)

with open(HIST_PATH) as f:
    h = json.load(f)

epochs     = range(1, len(h['accuracy']) + 1)
train_acc  = h['accuracy']
val_acc    = h['val_accuracy']
train_loss = h['loss']
val_loss   = h['val_loss']
best_epoch = int(np.argmax(val_acc)) + 1
best_val   = max(val_acc)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#1e1e2e')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ACCENT = '#89b4fa'
GREEN  = '#a6e3a1'
RED    = '#f38ba8'
YELLOW = '#f9e2af'
TEXT   = '#cdd6f4'
PANEL  = '#2a2a3e'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#45475a')
    ax.grid(True, alpha=0.2, color='#45475a')

# ── 1. Accuracy ───────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, [a*100 for a in train_acc], color=ACCENT, lw=2, label='Train')
ax1.plot(epochs, [a*100 for a in val_acc],   color=GREEN,  lw=2, label='Val')
ax1.axvline(best_epoch, color=YELLOW, lw=1.5, linestyle='--', alpha=0.8, label=f'Best epoch {best_epoch}')
ax1.axhline(best_val*100, color=GREEN, lw=1, linestyle=':', alpha=0.6)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
ax1.legend(facecolor='#313244', labelcolor=TEXT, framealpha=0.8)
style_ax(ax1, 'Accuracy')

# ── 2. Loss ───────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, train_loss, color=ACCENT, lw=2, label='Train')
ax2.plot(epochs, val_loss,   color=RED,    lw=2, label='Val')
ax2.axvline(best_epoch, color=YELLOW, lw=1.5, linestyle='--', alpha=0.8, label=f'Best epoch {best_epoch}')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(facecolor='#313244', labelcolor=TEXT, framealpha=0.8)
style_ax(ax2, 'Loss')

# ── 3. Train vs Val accuracy gap ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
gap = [abs(t - v)*100 for t, v in zip(train_acc, val_acc)]
ax3.fill_between(epochs, gap, alpha=0.4, color=YELLOW)
ax3.plot(epochs, gap, color=YELLOW, lw=2)
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Gap (%)')
style_ax(ax3, 'Train−Val Accuracy Gap (lower = less overfitting)')

# ── 4. Summary stats ──────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
ax4.axis('off')
style_ax(ax4, 'Training Summary')

final_gap = abs(train_acc[-1] - val_acc[-1]) * 100
stats = [
    ("Total epochs",        f"{len(epochs)}"),
    ("Best epoch",          f"{best_epoch}"),
    ("Best val accuracy",   f"{best_val*100:.2f}%"),
    ("Final train acc",     f"{train_acc[-1]*100:.2f}%"),
    ("Final val acc",       f"{val_acc[-1]*100:.2f}%"),
    ("Train/Val gap",       f"{final_gap:.2f}%"),
    ("Min val loss",        f"{min(val_loss):.4f}"),
]
for i, (label, value) in enumerate(stats):
    y = 0.88 - i * 0.12
    ax4.text(0.05, y, label + ":", transform=ax4.transAxes,
             color='#6c7086', fontsize=11, va='top')
    ax4.text(0.55, y, value, transform=ax4.transAxes,
             color=GREEN if 'acc' in label.lower() else TEXT,
             fontsize=11, fontweight='bold', va='top')

fig.suptitle('Sign Language MLP — Training Results',
             color=ACCENT, fontsize=16, fontweight='bold', y=0.98)

plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Graph saved → {OUT_PATH}")
