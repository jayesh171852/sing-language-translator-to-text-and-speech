"""
generate_performance_comparison.py
Grouped bar chart: Accuracy, Precision, F1-Score for each model.
Matches the style of the reference image.
Run: python generate_performance_comparison.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

# ── Data ──────────────────────────────────────────────────────────────────────
# Models from literature review + our model
# Accuracy, Precision, F1-Score (0.0 – 1.0 scale)
models = [
    'Logistic\nRegression',
    'SVC',
    'Deep\nLearning\n(CNN)',
    'Proposed\nModel\n(Landmark MLP)',
]

accuracy  = [0.79, 0.80, 0.81, 0.9954]
precision = [0.77, 0.75, 0.82, 0.9954]
f1_score  = [0.79, 0.83, 0.82, 0.9950]

# ── Plot ──────────────────────────────────────────────────────────────────────
x     = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Bars — match reference colors: orange, blue, green
bars1 = ax.bar(x - width,     accuracy,  width, label='Accuracy',  color='#E8A020', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x,             precision, width, label='Precision', color='#4BACC6', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width,     f1_score,  width, label='F1-Score',  color='#2EAA6E', edgecolor='white', linewidth=0.5)

# Value labels on top of each bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, color='#333333')
ax.set_ylim(0.2, 1.12)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, color='#333333')
ax.yaxis.grid(True, linestyle='-', alpha=0.4, color='#cccccc')
ax.set_axisbelow(True)

# Remove top and right spines (clean look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# Legend — top left like reference
ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
          facecolor='white', edgecolor='#cccccc',
          handlelength=1.5, handleheight=1.0)

# Title
ax.set_title('Performance Comparison', fontsize=14, fontweight='bold',
             color='#222222', pad=12)

# Caption below (like reference image)
fig.text(0.5, -0.02, 'Fig. Performance Comparison of Models',
         ha='center', fontsize=13, fontweight='bold', color='#222222')

plt.tight_layout()
path = os.path.join(OUT_DIR, 'performance_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path}")
