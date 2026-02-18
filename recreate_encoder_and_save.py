# recreate_encoder_and_save.py
import os, pickle, joblib
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "landmark_data"  # adjust if your folder is different
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# collect labels from subfolders or from filenames (fallback to A..Z)
labels = []
for name in sorted(os.listdir(DATA_DIR)):
    full = os.path.join(DATA_DIR, name)
    if os.path.isdir(full):
        labels.append(name)
# fallback: if no subfolders, try filenames without extension
if not labels:
    for f in sorted(os.listdir(DATA_DIR)):
        if f.lower().endswith(('.csv','.xlsx','.npy')):
            labels.append(os.path.splitext(f)[0].upper())

labels = sorted(list(dict.fromkeys(labels)))  # unique & stable order
print("Detected labels:", labels)

le = LabelEncoder()
le.fit(labels)

pickle_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
joblib_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
# remove old files if any
for p in [pickle_path, joblib_path]:
    try:
        if os.path.exists(p):
            os.remove(p)
    except:
        pass

# save both ways
with open(pickle_path, "wb") as f:
    pickle.dump(le, f)
joblib.dump(le, joblib_path)

print("Saved new encoder:", pickle_path, "and", joblib_path)
print("Classes:", le.classes_)
