# train_mlp_fix.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# === Configuration ===
DATA_DIR = "landmark_data"
MODEL_PATH = "models/final_landmark_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"

# Helper to load a single file and return numeric matrix
def load_table_numeric(path):
    # read with pandas (works for csv or xlsx)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # keep only numeric columns (drop any text columns)
    df_num = df.select_dtypes(include=[np.number])
    # If there are no numeric columns, try converting all to numeric (coerce)
    if df_num.shape[1] == 0:
        df = df.apply(pd.to_numeric, errors="coerce")
        df_num = df.select_dtypes(include=[np.number])
    # fill NaNs (you can choose method: fill with 0 or forward/backward fill)
    df_num = df_num.fillna(0)
    return df_num.values.astype("float32")

# === Step 1: Load all data and detect column consistency ===
X_list = []
y_list = []
col_counts = {}
for file in sorted(os.listdir(DATA_DIR)):
    if file.lower().endswith(('.csv', '.xlsx')):
        label = os.path.splitext(file)[0].strip().upper()
        path = os.path.join(DATA_DIR, file)
        try:
            arr = load_table_numeric(path)  # numpy array (n_rows, n_numeric_cols)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            raise

        if arr.ndim == 1:
            arr = arr.reshape(-1, arr.shape[0])  # ensure 2D
        X_list.append(arr)
        y_list += [label] * arr.shape[0]
        col_counts[arr.shape[1]] = col_counts.get(arr.shape[1], 0) + arr.shape[0]
        print(f"Loaded {file} -> samples: {arr.shape[0]}, cols: {arr.shape[1]}")

# Quick sanity checks
if len(X_list) == 0:
    raise SystemExit("No data files found in landmark_data/ (csv or xlsx).")

# Ensure all files have same number of columns
unique_col_counts = sorted(col_counts.keys())
print("Column counts present across files:", col_counts)
if len(unique_col_counts) != 1:
    print("Warning: multiple different numeric column counts detected:", unique_col_counts)
    # try to handle the most common column count
    most_common_count = max(col_counts.items(), key=lambda x: x[1])[0]
    print("Will attempt to trim/pad arrays to the most common column count:", most_common_count)
    # trim or pad each array to most_common_count
    X_normed = []
    for arr in X_list:
        c = arr.shape[1]
        if c > most_common_count:
            X_normed.append(arr[:, :most_common_count])
        elif c < most_common_count:
            # pad with zeros on the right
            pad = np.zeros((arr.shape[0], most_common_count - c), dtype="float32")
            X_normed.append(np.concatenate([arr, pad], axis=1))
        else:
            X_normed.append(arr)
    X = np.vstack(X_normed)
else:
    X = np.vstack(X_list)

y = np.array(y_list)
print(f"Total samples: {X.shape}, Total labels: {len(np.unique(y))}")

# === Step 2: Encode labels ===
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

# === Step 3: Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
print("Classes:", list(le.classes_))

# === Step 4: Define MLP model ===
input_dim = X.shape[1]
num_classes = y_cat.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 5: Train ===
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# === Step 6: Save model ===
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
joblib.dump(le, ENCODER_PATH)

print("\n✅ Model training complete!")
print(f"Saved model: {MODEL_PATH}")
print(f"Saved label encoder: {ENCODER_PATH}")
