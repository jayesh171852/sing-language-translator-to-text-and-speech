# GPU Training & Model Setup Notes

## Quick Verification
To run the live Sign Language recognition application:
```bash
python run_app.py
```
*(or on Windows, double-click `run_app.bat`)*

---

## Restoring Backup Model (If needed)
If you ever need to restore the original CNN model from backup:
```powershell
copy models\cnn8grps_rad1_model_backup.h5 models\cnn8grps_rad1_model.h5
```

---

## Setting Up GPU Training (NVIDIA RTX / CUDA)
1. **Python 3.11 Environment**:
   - Official Python 3.11 Installer: [Python 3.11.9 (64-bit)](https://www.python.org/downloads/release/python-3119/)
   - Ensure you check **"Add Python to PATH"** during installation.

2. **Why GPU Acceleration?**:
   - The landmark MLP (`models/landmark_model.keras`) currently achieves **97.96% validation accuracy** and is lightweight.
   - For full CNN retraining on image datasets (`data/processed` / `AtoZ_3.1`), training on CPU takes hours, whereas an NVIDIA RTX GPU (e.g., RTX 4050) completes epochs in ~10–15 minutes.
   - Python 3.11 is recommended for Windows native GPU TensorFlow support (or use WSL2 / Google Colab with `training/train_colab.ipynb`).

3. **Running Training**:
   - **Landmark MLP (Recommended, 97.96% acc)**:
     ```bash
     python training/train_landmark.py
     ```
   - **CNN Model Fine-tuning**:
     ```bash
     python training/train_cnn.py
     ```
   - **Google Colab Training**:
     Open `training/train_colab.ipynb` directly in Google Colab for free cloud GPU training.
