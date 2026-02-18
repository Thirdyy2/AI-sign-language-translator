# train_lstm.py (robust loader)
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Try to locate data files either in cwd or in models/ folder
CANDIDATE_DIRS = [".", "models", os.path.join(os.path.dirname(__file__), "models")]
def find_file(name):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

files = {
    "X_train": find_file("X_train_seq.npy"),
    "y_train": find_file("y_train_seq.npy"),
    "X_test": find_file("X_test_seq.npy"),
    "y_test": find_file("y_test_seq.npy")
}

print("Resolved paths:")
for k,v in files.items():
    print(f"  {k}: {v}")

# If any missing, show helpful error and exit
missing = [k for k,v in files.items() if v is None]
if missing:
    raise FileNotFoundError(f"Missing files: {missing}. Run prepare_seq_dataset.py or move the .npy files into project root or models/")

# Load arrays
X_train = np.load(files["X_train"])
y_train = np.load(files["y_train"])
X_test  = np.load(files["X_test"])
y_test  = np.load(files["y_test"])

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)

# Build LSTM model
timesteps = X_train.shape[1]
features = X_train.shape[2]
num_classes = len(np.unique(y_train))
print("Timesteps, features, classes:", timesteps, features, num_classes)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[es]
)

# save in models/ (create if needed)
out_dir = "models"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "sign_language_lstm.keras")
model.save(out_path)
print("Saved LSTM model to", out_path)
