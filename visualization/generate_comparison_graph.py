"""
generate_comparison_graph.py — Comparison graph of literature review models vs our model.
Run: python generate_comparison_graph.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Data from the literature review table ─────────────────────────────────────
# N/A and "High accuracy" entries are excluded from bar chart (no numeric value)
models = [
    ("Starner et al.\n(Computer Vision)",     90.0),
    ("Molchanov et al.\n(3D CNN)",             92.5),   # midpoint of 91-94%
    ("Pigou et al.\n(CNN + RNN)",              92.0),
    ("Camgoz et al.\n(Transformer)",           92.0),
    ("Ko et al.\n(Keypoint Estimation)",       90.0),
    ("Kumar & Singh\n(CNN + OpenCV)",          93.0),
    ("Our Model\n(Landmark MLP)",              97.96),
]

labels   = [m[0] for m in models]
accuracy = [m[1] for m in models]

# Colors — highlight our model
colors = [
    '#89b4fa','#89b4fa','#89b4fa',
    '#89b4fa','#89b4fa','#89b4fa',
    '#a6e3a1',   # our model — green
]

BG    = '#1e1e2e'
PANEL = '#2a2a3e'
TEXT  = '#cdd6f4'
SUB   = '#6c7086'
RED   = '#f38ba8'
GREEN = '#a6e3a1'
ACCENT= '#89b4fa'
YELLOW= '#f9e2af'

# ── FIG A: Horizontal Bar Chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

y_pos = np.arange(len(labels))
bars  = ax.barh(y_pos, accuracy, color=colors, edgecolor='#45475a',
                linewidth=0.8, height=0.6)

# Value labels on bars
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f'{acc:.1f}%', va='center', color=TEXT, fontsize=10, fontweight='bold')

# Reference line at 97.96%
ax.axvline(97.96, color=GREEN, lw=1.5, ls='--', alpha=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, color=TEXT, fontsize=9)
ax.set_xlabel('Accuracy (%)', color=TEXT, fontsize=11)
ax.set_xlim(85, 102)
ax.set_title('Accuracy Comparison: Literature Review vs Our Model',
             color=ACCENT, fontsize=13, fontweight='bold', pad=12)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values(): spine.set_edgecolor('#45475a')
ax.grid(True, axis='x', alpha=0.15, color='#45475a')

legend_patches = [
    mpatches.Patch(color=ACCENT, label='Existing Methods'),
    mpatches.Patch(color=GREEN,  label='Our Model (Landmark MLP)'),
]
ax.legend(handles=legend_patches, facecolor=PANEL, labelcolor=TEXT,
          framealpha=0.9, fontsize=10, loc='lower right')

plt.tight_layout()
path = os.path.join(OUT_DIR, 'fig_comparison_bar.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path}")


# ── FIG B: Full Table (all 10 papers + our model) ────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

columns = ['SR', 'Authors', 'Method', 'Summary', 'Accuracy']
rows = [
    ['1',  'Starner et al.',      'Computer Vision',       'First real-time sign language recognition system',          '~90%'],
    ['2',  'Ong & Ranganath',     'Survey',                'Identified challenges in segmentation & feature extraction', 'N/A'],
    ['3',  'Koller et al.',       'Statistical Models',    'Continuous sign language recognition for large vocabulary',  'High'],
    ['4',  'Molchanov et al.',    '3D CNN',                'Deep learning for dynamic gesture recognition',              '91–94%'],
    ['5',  'Pigou et al.',        'CNN + RNN',             'Improved gesture recognition using temporal modeling',       '~92%'],
    ['6',  'Camgoz et al.',       'Transformer Model',     'End-to-end sign language translation system',               '92%'],
    ['7',  'Huang et al.',        'Video-based Method',    'Removed need for temporal segmentation',                    'High'],
    ['8',  'Rastgoo et al.',      'Survey (Deep Learning)','Reviewed modern deep learning for sign language',           'N/A'],
    ['9',  'Ko et al.',           'Keypoint Estimation',   'Human skeletal tracking for gesture detection',             '~90%'],
    ['10', 'Kumar & Singh',       'CNN + OpenCV',          'Real-time sign language recognition system',                '~93%'],
    ['11', 'Ours',                'Landmark MLP',          'MediaPipe landmarks + normalized MLP, 26-class direct',     '97.96%'],
]

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width([0, 1, 2, 3, 4])

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(PANEL)
    cell.set_edgecolor('#45475a')
    cell.set_text_props(color=TEXT)
    if row == 0:
        cell.set_facecolor(ACCENT + '55')
        cell.set_text_props(color=TEXT, fontweight='bold')
    elif row == 11:   # our model row
        cell.set_facecolor(GREEN + '33')
        cell.set_text_props(color=GREEN, fontweight='bold')

ax.set_title('Literature Review Comparison Table',
             color=ACCENT, fontsize=14, fontweight='bold',
             pad=20, y=1.02)

plt.tight_layout()
path = os.path.join(OUT_DIR, 'fig_comparison_table_full.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path}")


# ── FIG C: Line/Dot chart showing progression ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

x      = np.arange(len(labels))
colors2 = [ACCENT]*6 + [GREEN]
sizes   = [80]*6 + [200]

ax.plot(x, accuracy, color=SUB, lw=1.5, ls='--', zorder=1)
sc = ax.scatter(x, accuracy, c=colors2, s=sizes, zorder=2, edgecolors='white', linewidths=0.8)

for i, (xi, acc) in enumerate(zip(x, accuracy)):
    offset = 0.8 if i < len(accuracy)-1 else -0.8
    ax.text(xi, acc + offset, f'{acc:.1f}%',
            ha='center', color=TEXT if i < len(accuracy)-1 else GREEN,
            fontsize=9, fontweight='bold' if i == len(accuracy)-1 else 'normal')

ax.axhline(97.96, color=GREEN, lw=1.2, ls=':', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([m[0].split('\n')[0] for m in models], rotation=20, ha='right',
                   color=TEXT, fontsize=9)
ax.set_ylabel('Accuracy (%)', color=TEXT, fontsize=11)
ax.set_ylim(85, 102)
ax.set_title('Model Accuracy Progression — Literature vs Our Approach',
             color=ACCENT, fontsize=13, fontweight='bold', pad=12)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values(): spine.set_edgecolor('#45475a')
ax.grid(True, alpha=0.15, color='#45475a')

legend_patches = [
    mpatches.Patch(color=ACCENT, label='Existing Methods'),
    mpatches.Patch(color=GREEN,  label='Our Model — 97.96%'),
]
ax.legend(handles=legend_patches, facecolor=PANEL, labelcolor=TEXT,
          framealpha=0.9, fontsize=10)

plt.tight_layout()
path = os.path.join(OUT_DIR, 'fig_comparison_line.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved: {path}")

print("\nAll comparison figures saved to paper_figures/")
print("  fig_comparison_bar.png   — horizontal bar chart")
print("  fig_comparison_table_full.png — full table with all 11 entries")
print("  fig_comparison_line.png  — dot/line progression chart")
