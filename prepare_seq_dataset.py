# prepare_seq_dataset.py  -- improved, robust, recursive search + diagnostics
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Config
SEQ_DIR = "seq_data"    # change if your folder name differs
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional override: set EXPECTED_SHAPE = (TIMESTEPS, 63)
# If None, script will infer the most-common shape from files.
EXPECTED_SHAPE = None   # e.g. (30,63) or (16,63) ; set to None to auto-detect

# Find all .npy files recursively
npy_files = []
for root, dirs, files in os.walk(SEQ_DIR):
    for f in files:
        if f.lower().endswith(".npy"):
            npy_files.append(os.path.join(root, f))

print("Searching for .npy files under:", os.path.abspath(SEQ_DIR))
print("Found", len(npy_files), "npy files (first 20 shown):")
for p in npy_files[:20]:
    print("  ", p)
if len(npy_files) == 0:
    print("ERROR: No .npy files found. Make sure you run this from the project root and SEQ_DIR is correct.")
    raise SystemExit(1)

# Load shapes and group by label (label = direct parent folder name)
files_by_label = defaultdict(list)
shapes_count = defaultdict(int)
shapes_example = {}

for p in npy_files:
    try:
        arr = np.load(p)
    except Exception as e:
        print("Failed to load", p, ":", e)
        continue
    shapes_count[arr.shape] += 1
    shapes_example[arr.shape] = p
    # label is folder name immediate parent of file
    parent = os.path.basename(os.path.dirname(p))
    files_by_label[parent].append((p, arr))

print("\nDetected sequence shapes and counts:")
for shp, cnt in sorted(shapes_count.items(), key=lambda x: (-x[1], x[0])):
    print(" ", shp, " ->", cnt, "files (example:", shapes_example[shp], ")")

# Decide expected shape
if EXPECTED_SHAPE is None:
    # choose most common shape
    most_common_shape = max(shapes_count.items(), key=lambda x: x[1])[0]
    EXPECTED_SHAPE = most_common_shape
    print("\nAuto-detected EXPECTED_SHAPE =", EXPECTED_SHAPE)
else:
    print("\nUser-specified EXPECTED_SHAPE =", EXPECTED_SHAPE)

timesteps_expected, feats_expected = EXPECTED_SHAPE

# Normalize (trim/pad) sequences and build arrays and labels
X_list = []
y_list = []
skipped = 0

for label, arr_list in sorted(files_by_label.items()):
    print(f"\nProcessing label '{label}' -> {len(arr_list)} files")
    for p, arr in arr_list:
        # ensure 2D array
        arr = np.array(arr, dtype="float32")
        if arr.ndim == 1:
            # try to reshape if length matches timesteps*feats
            if arr.size == timesteps_expected * feats_expected:
                arr = arr.reshape(timesteps_expected, feats_expected)
            else:
                print("  Skipping (1D with unexpected size):", p, arr.shape)
                skipped += 1
                continue
        # If shape matches, accept
        if arr.shape == (timesteps_expected, feats_expected):
            X_list.append(arr)
            y_list.append(label)
        else:
            # If timesteps differ, try trim or pad timesteps
            t, f = arr.shape
            if f != feats_expected:
                # If features mismatch, try trimming or padding features
                if f > feats_expected:
                    arr = arr[:, :feats_expected]
                else:
                    pad = np.zeros((t, feats_expected - f), dtype="float32")
                    arr = np.concatenate([arr, pad], axis=1)
                f = arr.shape[1]
            # Now adjust timesteps
            if t > timesteps_expected:
                arr = arr[:timesteps_expected, :]
                X_list.append(arr)
                y_list.append(label)
            elif t < timesteps_expected:
                # pad extra frames by repeating last frame
                if t == 0:
                    print("  Skipping (empty array):", p)
                    skipped += 1
                    continue
                last = np.repeat(arr[-1:, :], timesteps_expected - t, axis=0)
                arr2 = np.concatenate([arr, last], axis=0)
                X_list.append(arr2)
                y_list.append(label)
            else:
                # final fallback
                X_list.append(arr)
                y_list.append(label)

print(f"\nTotal accepted sequences: {len(X_list)}  (skipped {skipped})")

if len(X_list) == 0:
    print("ERROR: After processing, no sequences available. Check your .npy files shapes.")
    raise SystemExit(1)

X = np.stack(X_list).astype("float32")
y = np.array(y_list)

print("Final X shape:", X.shape)
# Encode labels to integers
labels = sorted(list(set(y)))
label_to_index = {lbl: i for i, lbl in enumerate(labels)}
y_int = np.array([label_to_index[l] for l in y], dtype="int32")

# Save label map for training later
np.save(os.path.join(OUT_DIR, "labels_seq.npy"), np.array(labels))

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=42, stratify=y_int)

print("Train/test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Save arrays
np.save(os.path.join(OUT_DIR, "X_train_seq.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train_seq.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_test_seq.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_test_seq.npy"), y_test)

print("\nSaved X_train_seq.npy etc. to", os.path.abspath(OUT_DIR))
print("Labels:", labels)
