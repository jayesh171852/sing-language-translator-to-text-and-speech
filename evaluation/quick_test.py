"""Quick accuracy test on the current model using the full dataset."""
import os, numpy as np, cv2
from tensorflow.keras.models import load_model

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    model_path = config.CNN_MODEL_PATH
    DATA_DIR = config.PROCESSED_DATA_DIR
except Exception:
    model_path = os.path.join(PROJECT_ROOT, 'models', 'cnn8grps_rad1_model.h5')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

model = load_model(model_path)
print(f"Model input: {model.input_shape}")

LETTER_TO_GROUP = {
    'A':0,'E':0,'M':0,'N':0,'S':0,'T':0,
    'B':1,'D':1,'F':1,'I':1,'K':1,'R':1,'U':1,'V':1,'W':1,
    'C':2,'O':2,'G':3,'H':3,'L':4,'P':5,'Q':5,'Z':5,'X':6,'Y':7,'J':7,
}
GROUP_NAMES = {0:'AEMNST',1:'BDFIKRUVW',2:'CO',3:'GH',4:'L',5:'PQZ',6:'X',7:'YJ'}

correct = total = 0
group_correct = {i:0 for i in range(8)}
group_total   = {i:0 for i in range(8)}

# DATA_DIR resolved above
for letter, group in LETTER_TO_GROUP.items():
    d = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(d): continue
    files = [f for f in os.listdir(d) if f.endswith('.jpg')][:50]  # 50 per letter
    batch = []
    for f in files:
        img = cv2.imread(os.path.join(d, f))
        if img is None: continue
        if img.shape[:2] != (400,400): img = cv2.resize(img,(400,400))
        batch.append(img.astype('float32')/255.0)
    if not batch: continue
    preds = np.argmax(model.predict(np.array(batch), verbose=0), axis=1)
    c = int(np.sum(preds == group))
    group_correct[group] += c
    group_total[group]   += len(batch)
    correct += c
    total   += len(batch)
    print(f"  {letter}: {c}/{len(batch)} ({c/len(batch)*100:.0f}%) → group {group}")

print("\n── Group Summary ──────────────────────")
for g in range(8):
    if group_total[g] == 0: continue
    acc = group_correct[g]/group_total[g]*100
    bar = '█'*int(acc/5) + '░'*(20-int(acc/5))
    print(f"  Group {g} [{bar}] {acc:.1f}%  ({GROUP_NAMES[g]})")

overall = correct/total*100 if total else 0
print(f"\n  OVERALL: {overall:.1f}%  ({correct}/{total})")
