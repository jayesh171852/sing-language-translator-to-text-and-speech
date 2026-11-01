"""
generate_workflow.py — Workflow chart matching the reference style.
Run: python generate_workflow.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    OUT_DIR = config.PAPER_FIGURES_DIR
except Exception:
    OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Steps ─────────────────────────────────────────────────────────────────────
steps = [
    ("Data Collection",
     "Load 87,000 real ASL hand photos from Kaggle dataset (A-Z, 3000 per letter)."),

    ("Feature Extraction",
     "Use MediaPipe to extract 21 hand landmark points (x, y, z) = 63 features per image."),

    ("Normalization",
     "Translate wrist to origin and scale by max distance — removes size and background dependency."),

    ("Train-Test Split",
     "Split data into 85% training and 15% validation sets (stratified by class)."),

    ("Model Training",
     "Train 26-class MLP: Dense(512) → Dense(256) → Dense(128) → Dense(64) → Dense(26, Softmax)."),

    ("Model Evaluation",
     "Evaluate using accuracy, precision, F1-score and confusion matrix on test set."),

    ("Real-Time Inference",
     "Webcam → MediaPipe → Normalize → MLP → Confidence voting (8/10 frames) → Letter."),

    ("Final Output",
     "Recognized letters form words and sentences, converted to speech via Text-to-Speech."),
]

# ── Layout ────────────────────────────────────────────────────────────────────
fig_h = len(steps) * 1.35 + 0.5
fig, ax = plt.subplots(figsize=(7, fig_h))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, fig_h)

box_w   = 8.5
box_h   = 0.95
x_left  = 0.75
gap     = 1.35   # vertical spacing between box tops

# Color gradient from light blue to slightly deeper blue (like reference)
colors = [
    '#AED6F1', '#85C1E9', '#7FB3D3', '#76B4D4',
    '#6DAED0', '#64A8CC', '#5BA2C8', '#529CC4',
]

for i, (title, subtitle) in enumerate(steps):
    y_top = fig_h - 0.3 - i * gap

    # Box
    color = colors[i % len(colors)]
    rect = FancyBboxPatch(
        (x_left, y_top - box_h), box_w, box_h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor='white',
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(rect)

    # Bold title
    ax.text(x_left + box_w/2, y_top - box_h*0.32,
            title,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#1a1a2e', zorder=3)

    # Subtitle
    ax.text(x_left + box_w/2, y_top - box_h*0.68,
            subtitle,
            ha='center', va='center',
            fontsize=8.5, color='#2c3e50',
            style='italic', zorder=3,
            wrap=True)

    # Arrow to next box
    if i < len(steps) - 1:
        arrow_y_start = y_top - box_h - 0.01
        arrow_y_end   = y_top - gap + 0.01
        ax.annotate('',
                    xy=(x_left + box_w/2, arrow_y_end),
                    xytext=(x_left + box_w/2, arrow_y_start),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color='#555555',
                        lw=1.8,
                        mutation_scale=14,
                    ),
                    zorder=4)

plt.tight_layout(pad=0.3)
path = os.path.join(OUT_DIR, 'workflow_chart.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path}")
