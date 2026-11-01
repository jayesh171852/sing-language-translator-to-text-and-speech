"""
generate_infographic.py — Semicircle infographic for Sign Language project.
Run: python generate_infographic.py
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

fig, ax = plt.subplots(figsize=(8, 9))
fig.patch.set_facecolor('#FDFAF4')
ax.set_facecolor('#FDFAF4')
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.1, 2.2)
ax.axis('off')
ax.set_aspect('equal')

# ── Semicircle segments ───────────────────────────────────────────────────────
# 6 segments across 180 degrees
segments = [
    (0,   30,  '#E74C3C'),   # red
    (30,  60,  '#E67E22'),   # orange
    (60,  90,  '#F1C40F'),   # yellow
    (90,  120, '#A8C640'),   # yellow-green
    (120, 150, '#5DBB63'),   # green
    (150, 180, '#27AE60'),   # dark green
]

outer_r = 1.3
inner_r = 0.65

for start_deg, end_deg, color in segments:
    start_rad = np.radians(start_deg)
    end_rad   = np.radians(end_deg)
    theta     = np.linspace(start_rad, end_rad, 50)

    # Outer arc
    x_outer = outer_r * np.cos(theta)
    y_outer = outer_r * np.sin(theta)
    # Inner arc (reversed)
    x_inner = inner_r * np.cos(theta[::-1])
    y_inner = inner_r * np.sin(theta[::-1])

    xs = np.concatenate([x_outer, x_inner])
    ys = np.concatenate([y_outer, y_inner])

    ax.fill(xs, ys, color=color, zorder=2)
    # White gap between segments
    ax.plot([inner_r * np.cos(start_rad), outer_r * np.cos(start_rad)],
            [inner_r * np.sin(start_rad), outer_r * np.sin(start_rad)],
            color='#FDFAF4', lw=2.5, zorder=3)

# Close last gap
ax.plot([inner_r * np.cos(np.radians(180)), outer_r * np.cos(np.radians(180))],
        [inner_r * np.sin(np.radians(180)), outer_r * np.sin(np.radians(180))],
        color='#FDFAF4', lw=2.5, zorder=3)
ax.plot([inner_r * np.cos(0), outer_r * np.cos(0)],
        [inner_r * np.sin(0), outer_r * np.sin(0)],
        color='#FDFAF4', lw=2.5, zorder=3)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(0, 2.05, 'SIGN LANGUAGE',
        ha='center', va='center',
        fontsize=22, fontweight='bold', color='#1a2a4a',
        fontfamily='DejaVu Sans')
ax.text(0, 1.78, 'TO TEXT & SPEECH',
        ha='center', va='center',
        fontsize=16, fontweight='bold', color='#1a2a4a')

# ── Bottom icons + labels ─────────────────────────────────────────────────────
# 4 items: left-bottom, center-left, center-right, right-bottom
items = [
    (-1.35, -0.55, '#E74C3C', 'HAND\nDETECTION'),
    (-0.42, -0.75, '#E67E22', 'LANDMARK\nEXTRACTION'),
    ( 0.42, -0.75, '#27AE60', 'MLP\nCLASSIFIER'),
    ( 1.35, -0.55, '#2980B9', 'TEXT TO\nSPEECH'),
]

icon_symbols = ['H', 'L', 'M', 'S']   # H=Hand, L=Landmark, M=MLP, S=Speech

for (x, y, color, label), sym in zip(items, icon_symbols):
    circle = plt.Circle((x, y+0.18), 0.22,
                         color=color, zorder=4, alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y+0.18, sym,
            ha='center', va='center', fontsize=14,
            fontweight='bold', color='white', zorder=5)
    ax.text(x, y-0.12, label,
            ha='center', va='top',
            fontsize=9, fontweight='bold', color='#1a2a4a',
            linespacing=1.4)

# Accuracy badge in center of semicircle
ax.text(0, 0.1, '99.54%',
        ha='center', va='center',
        fontsize=15, fontweight='bold', color='#1a2a4a')
ax.text(0, -0.15, 'ACCURACY',
        ha='center', va='center',
        fontsize=8, color='#555555', fontweight='bold')

plt.tight_layout(pad=0.2)
path = os.path.join(OUT_DIR, 'project_infographic.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#FDFAF4')
plt.close()
print(f"Saved: {path}")
