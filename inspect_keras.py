# inspect_keras.py
import os, zipfile, traceback
from tensorflow.keras.models import load_model

model_path = r"C:\Users\Tanya\Desktop\sign_language_project\models\sign_language_lstm.keras"
outdir = r"C:\Users\Tanya\Desktop\sign_language_project\models\sign_language_lstm_extracted"

print("Path:", model_path)
print("Exists:", os.path.exists(model_path))
print("Is file:", os.path.isfile(model_path))
if os.path.exists(model_path):
    print("Size (bytes):", os.path.getsize(model_path))

print("\nIs zipfile according to zipfile.is_zipfile():", zipfile.is_zipfile(model_path))

if zipfile.is_zipfile(model_path):
    print("\n--- Listing zip contents (first 50 entries) ---")
    with zipfile.ZipFile(model_path, 'r') as z:
        names = z.namelist()
        for i, n in enumerate(names[:50], 1):
            print(f"{i:03d}: {n}")
        print("... total entries:", len(names))

    # extract to outdir
    try:
        print("\nExtracting to:", outdir)
        with zipfile.ZipFile(model_path, 'r') as z:
            z.extractall(outdir)
        print("Extraction done.")
        # show some extracted items
        for root, dirs, files in os.walk(outdir):
            print("Extracted folder sample:", root)
            print(" dirs:", dirs[:10])
            print(" files:", files[:10])
            break

        # Try loading from the extracted folder (many .keras archives contain a saved_model folder)
        try:
            print("\nAttempting to load model from extracted folder with load_model(outdir)...")
            m = load_model(outdir)
            print("Loaded model from extracted folder. input_shape:", getattr(m, "input_shape", None))
            save_h5 = os.path.join(os.path.dirname(model_path), "sign_language_lstm_from_extracted.h5")
            m.save(save_h5)
            print("Saved HDF5 to:", save_h5)
        except Exception as e:
            print("Loading extracted folder failed:")
            traceback.print_exc()
    except Exception:
        print("Extraction failed:")
        traceback.print_exc()
else:
    print("\nNot a zipfile. It may be corrupted or not a .keras archive.")
    # Try to open as HDF5 with h5py (some models are .h5 but misnamed)
    try:
        import h5py
        print("\nTrying to open as HDF5 via h5py...")
        with h5py.File(model_path, 'r') as f:
            print("HDF5 root keys:", list(f.keys())[:30])
            print("Looks like a valid HDF5 file.")
    except Exception:
        print("h5py open failed or not an HDF5 file:")
        traceback.print_exc()
