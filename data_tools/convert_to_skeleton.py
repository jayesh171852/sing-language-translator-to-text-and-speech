"""
convert_to_skeleton.py — Converts raw hand photo dataset to skeleton images.

Usage:
    python convert_to_skeleton.py --input "path/to/asl_alphabet_train" --output "AtoZ_3.1"

The input folder should have subfolders named A-Z, each containing images.
Output goes into your existing AtoZ_3.1 folder (adds to existing images).

Example:
    python convert_to_skeleton.py --input "C:/Downloads/asl_alphabet_train" --output "AtoZ_3.1"
"""

import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    MODEL_PATH = config.LM_TASK_PATH
    WHITE_PATH = config.WHITE_PATH
except Exception:
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    WHITE_PATH = os.path.join(PROJECT_ROOT, 'assets', 'white.jpg')
CANVAS_SIZE = 400

# Generate white canvas if missing
if not os.path.exists(WHITE_PATH):
    cv2.imwrite(WHITE_PATH, np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8) * 255)

# Only process letters A-Z (skip 'space', 'del', 'nothing' etc.)
VALID_LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# ── Setup MediaPipe hand detector ─────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(options)


def image_to_skeleton(img_bgr):
    """Convert a BGR hand photo to a 400x400 skeleton drawing. Returns None if no hand found."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand_lms = result.hand_landmarks[0]
    lm_list = [[int(lm.x * w), int(lm.y * h)] for lm in hand_lms]

    # Get bounding box of landmarks
    xs = [p[0] for p in lm_list]
    ys = [p[1] for p in lm_list]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bw = xmax - xmin
    bh = ymax - ymin

    if bw == 0 or bh == 0:
        return None

    # Center in canvas
    ox = (CANVAS_SIZE - bw) // 2
    oy = (CANVAS_SIZE - bh) // 2

    white = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8) * 255

    def pt(i):
        return (
            max(0, min(CANVAS_SIZE - 1, lm_list[i][0] - xmin + ox)),
            max(0, min(CANVAS_SIZE - 1, lm_list[i][1] - ymin + oy))
        )

    # Draw finger segments
    connections = (
        list(range(0, 4)) + list(range(5, 8)) +
        list(range(9, 12)) + list(range(13, 16)) +
        list(range(17, 20))
    )
    for t in connections:
        cv2.line(white, pt(t), pt(t + 1), (0, 255, 0), 3)

    # Draw palm connections
    for a, b in [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]:
        cv2.line(white, pt(a), pt(b), (0, 255, 0), 3)

    # Draw landmark dots
    for i in range(21):
        cv2.circle(white, pt(i), 4, (0, 0, 255), -1)

    return white


def convert_dataset(input_dir, output_dir):
    input_dir  = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_dir):
        print(f"ERROR: Input folder not found: {input_dir}")
        return

    letters = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.upper() in VALID_LETTERS
    ])

    if not letters:
        print(f"ERROR: No A-Z subfolders found in {input_dir}")
        print(f"Found: {os.listdir(input_dir)[:10]}")
        return

    print(f"Found {len(letters)} letters: {letters}")
    print(f"Output → {output_dir}\n")

    total_saved  = 0
    total_failed = 0

    for letter in letters:
        letter_up  = letter.upper()
        src_dir    = os.path.join(input_dir, letter)
        dst_dir    = os.path.join(output_dir, letter_up)
        os.makedirs(dst_dir, exist_ok=True)

        # Start count from existing images
        existing = len([f for f in os.listdir(dst_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        count = existing

        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        saved = failed = 0
        for fname in files:
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                failed += 1
                continue

            skeleton = image_to_skeleton(img)
            if skeleton is None:
                failed += 1
                continue

            out_path = os.path.join(dst_dir, f"{count}.jpg")
            cv2.imwrite(out_path, skeleton)
            count += 1
            saved += 1

        total_saved  += saved
        total_failed += failed
        pct = (saved / len(files) * 100) if files else 0
        print(f"  {letter_up}: {saved}/{len(files)} converted ({pct:.0f}%)  "
              f"[{failed} no hand detected]  total in folder: {count}")

    print(f"\n{'='*50}")
    print(f"  Done!  Saved: {total_saved}  |  Failed: {total_failed}")
    print(f"  Output folder: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True,
                        help='Path to dataset folder with A-Z subfolders')
    parser.add_argument('--output', default='data/processed',
                        help='Output folder (default: AtoZ_3.1)')
    args = parser.parse_args()
    convert_dataset(args.input, args.output)
