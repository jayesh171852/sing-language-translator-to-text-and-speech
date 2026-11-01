"""
config.py — Centralized Configuration and Path Management
Provides standardized, cross-platform paths for models, datasets, assets, and outputs.
"""
import os
import sys

# Project Root Directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Core Directories
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")
DATA_TOOLS_DIR = os.path.join(PROJECT_ROOT, "data_tools")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualization")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PAPER_FIGURES_DIR = os.path.join(OUTPUTS_DIR, "paper_figures")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Dataset Paths (with legacy fallbacks)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "asl_alphabet_train", "asl_alphabet_train")
if not os.path.exists(RAW_DATA_DIR):
    # Check alternate extracted structures
    alt_raw = os.path.join(DATA_DIR, "raw")
    if os.path.exists(alt_raw):
        RAW_DATA_DIR = alt_raw

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model Asset Paths
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "landmark_model.keras")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
LM_TASK_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn8grps_rad1_model.h5")
CNN_BACKUP_PATH = os.path.join(MODELS_DIR, "cnn8grps_rad1_model_backup.h5")
CNN_128_MODEL_PATH = os.path.join(MODELS_DIR, "cnn8grps_128_model.h5")

# Asset Paths
WHITE_PATH = os.path.join(ASSETS_DIR, "white.jpg")

# Add src to sys.path so modules can import from src cleanly
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_path(filename, search_dirs=None):
    """
    Finds a file by searching standard project directories.
    Returns the resolved absolute path or the default fallback.
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    if search_dirs is None:
        search_dirs = [PROJECT_ROOT, MODELS_DIR, ASSETS_DIR, DATA_DIR, OUTPUTS_DIR]

    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate

    return os.path.join(PROJECT_ROOT, filename)


def ensure_dirs():
    """Ensure all runtime directories exist."""
    for d in [MODELS_DIR, ASSETS_DIR, OUTPUTS_DIR, PAPER_FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


ensure_dirs()
