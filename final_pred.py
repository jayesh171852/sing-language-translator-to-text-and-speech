"""
Root entrypoint proxy for the Sign Language to Text & Speech application.
Allows running `python final_pred.py` directly from the project root.
"""
import sys
import os

# Ensure project root and src/ are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from src.final_pred import Application

if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
