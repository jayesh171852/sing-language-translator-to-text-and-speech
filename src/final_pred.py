"""
final_pred.py — Sign Language to Text & Speech
Uses a 26-class MLP trained on normalized MediaPipe hand landmarks.
"""
import numpy as np
import cv2
import os
import json
import traceback
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys

# Ensure local src/ and project root are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
for p in (BASE_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model
from hand_detector import HandDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    import config
    MODEL_PATH     = config.LANDMARK_MODEL_PATH
    LABEL_MAP_PATH = config.LABEL_MAP_PATH
    LM_TASK        = config.LM_TASK_PATH
    WHITE_PATH     = config.WHITE_PATH
except Exception:
    MODELS_DIR     = os.path.join(PROJECT_ROOT, 'models')
    ASSETS_DIR     = os.path.join(PROJECT_ROOT, 'assets')
    MODEL_PATH     = os.path.join(MODELS_DIR, 'landmark_model.keras') if os.path.exists(os.path.join(MODELS_DIR, 'landmark_model.keras')) else os.path.join(BASE_DIR, 'landmark_model.keras')
    LABEL_MAP_PATH = os.path.join(MODELS_DIR, 'label_map.json') if os.path.exists(os.path.join(MODELS_DIR, 'label_map.json')) else os.path.join(BASE_DIR, 'label_map.json')
    LM_TASK        = os.path.join(MODELS_DIR, 'hand_landmarker.task') if os.path.exists(os.path.join(MODELS_DIR, 'hand_landmarker.task')) else os.path.join(BASE_DIR, 'hand_landmarker.task')
    WHITE_PATH     = os.path.join(ASSETS_DIR, 'white.jpg') if os.path.exists(os.path.join(ASSETS_DIR, 'white.jpg')) else os.path.join(BASE_DIR, 'white.jpg')

USE_NEW_MODEL = os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH)
if not USE_NEW_MODEL:
    legacy_path = os.path.join(PROJECT_ROOT, 'models', 'cnn8grps_rad1_model.h5')
    MODEL_PATH = legacy_path if os.path.exists(legacy_path) else os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')
    print("WARNING: landmark_model.keras not found - using old CNN model.")
    print("Run  python training/train_landmark.py  for better accuracy.")

if not os.path.exists(WHITE_PATH):
    os.makedirs(os.path.dirname(WHITE_PATH), exist_ok=True)
    cv2.imwrite(WHITE_PATH, np.ones((400, 400, 3), np.uint8) * 255)

# ── Optional spell suggestions ────────────────────────────────────────────────
try:
    import enchant
    _dict = enchant.Dict("en-US")
    ENCHANT_OK = True
except Exception:
    ENCHANT_OK = False

# ── Theme ─────────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
PANEL_BG  = "#2a2a3e"
ACCENT    = "#89b4fa"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
TEXT      = "#cdd6f4"
SUBTEXT   = "#6c7086"
BTN_BG    = "#313244"
BTN_HOVER = "#45475a"
FONT_MONO = ("Consolas", 14)
FONT_BIG  = ("Consolas", 22, "bold")
FONT_SM   = ("Consolas", 11)

CONFIRM_THRESHOLD = 8
HISTORY_LEN       = 10

# ── Speech ────────────────────────────────────────────────────────────────────
def speak(text):
    text = text.strip().replace("'", "")
    if text:
        threading.Thread(
            target=lambda: os.system(
                f'powershell -Command "Add-Type -AssemblyName System.Speech; '
                f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$s.Rate = 1; $s.Speak(\\"{text}\\")"'
            ),
            daemon=True
        ).start()

# ── MediaPipe landmark extractor ──────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=LM_TASK)
lm_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE,
)
lm_detector = mp_vision.HandLandmarker.create_from_options(lm_options)

# HandDetector for skeleton drawing only
hd_draw = HandDetector(maxHands=1)


def get_landmarks(img_bgr):
    """
    Returns (feat_63d, raw_pixel_pts) or (None, None).
    feat: normalized 63-d vector — invariant to size/position/background.
    """
    h, w = img_bgr.shape[:2]
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm_detector.detect(mp_img)

    if not result.hand_landmarks:
        return None, None

    lms = result.hand_landmarks[0]
    raw = np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms], dtype=np.int32)
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float32)

    # Normalize: wrist to origin, scale by max distance
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        return None, None
    pts /= scale

    return pts.flatten(), raw


def draw_skeleton(raw_pts, size=320):
    """Draw skeleton on white canvas from raw pixel coords."""
    canvas = np.ones((size, size, 3), np.uint8) * 255
    xs, ys = raw_pts[:, 0], raw_pts[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    bw = max(xmax - xmin, 1)
    bh = max(ymax - ymin, 1)
    ox = (size - bw) // 2
    oy = (size - bh) // 2

    def pt(i):
        return (int(np.clip(raw_pts[i][0] - xmin + ox, 0, size-1)),
                int(np.clip(raw_pts[i][1] - ymin + oy, 0, size-1)))

    segs = list(range(0,4))+list(range(5,8))+list(range(9,12))+list(range(13,16))+list(range(17,20))
    for t in segs:
        cv2.line(canvas, pt(t), pt(t+1), (100, 220, 100), 2)
    for a, b in [(5,9),(9,13),(13,17),(0,5),(0,17)]:
        cv2.line(canvas, pt(a), pt(b), (100, 220, 100), 2)
    for i in range(21):
        cv2.circle(canvas, pt(i), 4, (60, 120, 255), -1)
    return canvas


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_suggestions(word):
    if not ENCHANT_OK or not word.strip():
        return ["", "", "", ""]
    try:
        s = _dict.suggest(word.strip())[:4]
        while len(s) < 4: s.append("")
        return s
    except Exception:
        return ["", "", "", ""]

def _hover_in(btn):  btn.config(bg=BTN_HOVER)
def _hover_out(btn): btn.config(bg=BTN_BG)

def styled_btn(parent, text, cmd, width=10):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=BTN_BG, fg=TEXT, activebackground=BTN_HOVER,
                  activeforeground=TEXT, relief="flat", bd=0,
                  font=FONT_MONO, width=width, cursor="hand2")
    b.bind("<Enter>", lambda e: _hover_in(b))
    b.bind("<Leave>", lambda e: _hover_out(b))
    return b


# ── Application ───────────────────────────────────────────────────────────────
class Application:

    def __init__(self):
        self.vs    = cv2.VideoCapture(0)
        self.model = load_model(MODEL_PATH)

        if USE_NEW_MODEL:
            with open(LABEL_MAP_PATH) as f:
                raw = json.load(f)
            self.label_map = {int(k): v for k, v in raw.items()}
        else:
            self.label_map = {}

        self.sentence    = ""
        self.word        = ""
        self.prev_char   = ""
        self.space_flag  = False
        self.next_flag   = True
        self.history     = [" "] * HISTORY_LEN
        self.suggestions = ["", "", "", ""]

        self._build_ui()
        # Use ASCII-friendly status messages to avoid UnicodeEncodeError on Windows terminals
        mode = "landmark MLP" if USE_NEW_MODEL else "legacy CNN (run train_landmark.py)"
        print(f"Model loaded [{mode}] - starting.")
        # Bring the Tkinter window to the foreground (Windows)
        self.root.after(100, lambda: self.root.attributes("-topmost", True))
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        # Start the video loop after the window is shown
        self.root.after(300, self.video_loop)

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language → Text & Speech")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1340x780")
        self.root.resizable(True, True)

        tk.Label(self.root, text="✋  Sign Language → Text & Speech",
                 bg=BG, fg=ACCENT, font=("Consolas", 20, "bold")).pack(pady=(10,0))

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=8)

        # Left — webcam
        left = tk.Frame(main, bg=PANEL_BG)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        tk.Label(left, text="Camera Feed", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SM).pack(pady=(6,2))
        self.panel_cam = tk.Label(left, bg="#000000")
        self.panel_cam.pack(padx=8, pady=4, fill="both", expand=True)
        self.status_var = tk.StringVar(value="⬤  No hand detected")
        self.status_lbl = tk.Label(left, textvariable=self.status_var,
                                   bg=PANEL_BG, fg=RED, font=FONT_SM)
        self.status_lbl.pack(pady=(2,6))

        # Right — skeleton + info
        right = tk.Frame(main, bg=BG)
        right.pack(side="right", fill="y", padx=(8,0))
        tk.Label(right, text="Hand Skeleton", bg=BG, fg=SUBTEXT, font=FONT_SM).pack(pady=(6,2))
        self.panel_skel = tk.Label(right, bg=PANEL_BG, width=320, height=320)
        self.panel_skel.pack(padx=4, pady=4)

        char_row = tk.Frame(right, bg=BG)
        char_row.pack(fill="x", pady=(8,2))
        tk.Label(char_row, text="Detected :", bg=BG, fg=SUBTEXT, font=FONT_SM).pack(side="left")
        self.char_var = tk.StringVar(value="—")
        tk.Label(char_row, textvariable=self.char_var,
                 bg=BG, fg=ACCENT, font=FONT_BIG).pack(side="left", padx=8)

        tk.Label(right, text="Confidence", bg=BG, fg=SUBTEXT, font=FONT_SM).pack(anchor="w")
        style = ttk.Style(); style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor=PANEL_BG, background=GREEN, thickness=14)
        self.conf_bar = ttk.Progressbar(right, style="green.Horizontal.TProgressbar",
                                        orient="horizontal", length=280,
                                        mode="determinate", maximum=HISTORY_LEN)
        self.conf_bar.pack(pady=(2,10))

        # Bottom
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=16, pady=(0,10))

        sent_frame = tk.Frame(bottom, bg=PANEL_BG)
        sent_frame.pack(fill="x", pady=(0,8))
        tk.Label(sent_frame, text="Sentence", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SM).pack(anchor="w", padx=8, pady=(4,0))
        self.sent_var = tk.StringVar(value="")
        tk.Label(sent_frame, textvariable=self.sent_var, bg=PANEL_BG, fg=TEXT,
                 font=("Consolas",16), wraplength=1280, justify="left", anchor="w").pack(fill="x", padx=12, pady=(0,8))

        sugg_frame = tk.Frame(bottom, bg=BG)
        sugg_frame.pack(fill="x", pady=(0,6))
        tk.Label(sugg_frame, text="Suggestions :", bg=BG, fg=SUBTEXT, font=FONT_SM).pack(side="left")
        self.sugg_btns = []
        for i in range(4):
            b = styled_btn(sugg_frame, "", lambda idx=i: self._apply_suggestion(idx), width=14)
            b.pack(side="left", padx=4)
            self.sugg_btns.append(b)

        act = tk.Frame(bottom, bg=BG)
        act.pack(fill="x")
        styled_btn(act, "🔊  Speak",    self.speak_fun,     width=12).pack(side="left", padx=4)
        styled_btn(act, "Space",         self.space_fun,     width=10).pack(side="left", padx=4)
        styled_btn(act, "Next Letter",   self.next_fun,      width=12).pack(side="left", padx=4)
        styled_btn(act, "⌫  Backspace", self.backspace_fun, width=14).pack(side="left", padx=4)
        styled_btn(act, "📋  Copy",      self.copy_fun,      width=12).pack(side="left", padx=4)
        styled_btn(act, "🗑  Clear",     self.clear_fun,     width=12).pack(side="left", padx=4)

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(10, self.video_loop)
                return

            frame = cv2.flip(frame, 1)

            # Webcam display
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb).resize((480,360)))
            self.panel_cam.imgtk = imgtk
            self.panel_cam.config(image=imgtk)

            feat, raw_pts = get_landmarks(frame)

            if feat is not None:
                self.status_var.set("⬤  Hand detected")
                self.status_lbl.config(fg=GREEN)

                # Draw skeleton
                skel   = draw_skeleton(raw_pts)
                imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(skel).resize((320,320), Image.LANCZOS))
                self.panel_skel.imgtk = imgtk2
                self.panel_skel.config(image=imgtk2)

                self.predict(feat)
            else:
                self.status_var.set("⬤  No hand detected")
                self.status_lbl.config(fg=RED)

            self.sent_var.set(self.sentence)

        except Exception:
            print("video_loop error:", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def predict(self, feat):
        prob = self.model.predict(feat.reshape(1,-1), verbose=0)[0]
        idx  = int(np.argmax(prob))
        ch   = self.label_map.get(idx, str(idx)) if USE_NEW_MODEL else str(idx)

        self.char_var.set(ch)
        self.history.pop(0)
        self.history.append(ch)
        count = self.history.count(ch)
        self.conf_bar["value"] = count

        if count >= CONFIRM_THRESHOLD:
            self._commit(ch)

    def _commit(self, ch):
        if ch == 'Space':
            if not self.space_flag:
                self.sentence += " "
                if self.word.strip(): speak(self.word)
                self.word = ""; self.space_flag = True; self.next_flag = False
        elif ch == 'Next':
            if not self.next_flag:
                self.next_flag = True; self.space_flag = False
        elif ch == 'Backspace':
            self.sentence = self.sentence[:-1] if self.sentence else ""
            self.space_flag = False; self.next_flag = False
        else:
            if self.next_flag and ch != self.prev_char:
                self.space_flag = False
                self.sentence  += str(ch)
                self.prev_char  = ch
                parts = self.sentence.strip().split()
                self.word = parts[-1] if parts else ""
                self.suggestions = get_suggestions(self.word)
                for i, b in enumerate(self.sugg_btns):
                    b.config(text=self.suggestions[i])

    def _apply_suggestion(self, idx):
        word = self.suggestions[idx].strip()
        if not word: return
        parts = self.sentence.rstrip().rsplit(" ", 1)
        self.sentence = (parts[0]+" " if len(parts)>1 else "") + word.upper() + " "
        self.sent_var.set(self.sentence)

    def speak_fun(self):
        if self.sentence.strip(): speak(self.sentence.strip())

    def space_fun(self):
        self.sentence += " "; self.word = ""
        self.space_flag = True; self.next_flag = False
        self.sent_var.set(self.sentence)

    def next_fun(self):
        self.next_flag = True; self.space_flag = False; self.prev_char = ""

    def backspace_fun(self):
        self.sentence = self.sentence[:-1] if self.sentence else ""
        self.sent_var.set(self.sentence)

    def copy_fun(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.sentence.strip())
        self.root.update()

    def clear_fun(self):
        self.sentence = ""; self.word = ""; self.prev_char = ""
        self.space_flag = False; self.next_flag = True
        self.history = [" "] * HISTORY_LEN
        self.suggestions = ["","","",""]
        self.sent_var.set(""); self.char_var.set("—")
        self.conf_bar["value"] = 0
        for b in self.sugg_btns: b.config(text="")

    def destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == '__main__':
    app = Application()
    app.root.mainloop()
