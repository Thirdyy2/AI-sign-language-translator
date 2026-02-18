# prepare_sequences.py
import glob, numpy as np, os, pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

files = glob.glob("seq_data/*/*.npy")
X_list, y_list = [], []
for f in files:
    arr = np.load(f)    # shape (TIMESTEPS, 63)
    X_list.append(arr)
    label = os.path.basename(os.path.dirname(f))
    y_list.append(label)

X = np.stack(X_list)     # (N, TIMESTEPS, 63)
y = np.array(y_list)
le = LabelEncoder()
y_enc = le.fit_transform(y)
pickle.dump(le, open("label_encoder_seq.pkl", "wb"))
print("Classes:", list(le.classes_))
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
np.save("X_train_seq.npy", X_train); np.save("X_val_seq.npy", X_val)
np.save("y_train_seq.npy", y_train); np.save("y_val_seq.npy", y_val)
print("Saved arrays:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)
