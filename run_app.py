"""
Sign Language to Text and Speech Conversion — Application Entrypoint
Run this file to start the real-time gesture recognition interface.
"""
import sys
import os

# Initialize configuration and paths
import config

if __name__ == "__main__":
    print("=" * 60)
    print("  Sign Language to Text and Speech Conversion")
    print("=" * 60)
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Model Path:   {config.LANDMARK_MODEL_PATH}")
    print("Launching user interface...\n")

    # Launch application
    from final_pred import Application
    app = Application()
    app.root.mainloop()
