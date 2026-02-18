# convert_using_train_code.py
import importlib.util
import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "sign_language_lstm.keras")   # your weights-only file
OUT_H5 = os.path.join(MODELS_DIR, "sign_language_lstm.h5")

print("Project root:", PROJECT_ROOT)
print("Looking for train_lstm.py in project root...")

train_module_path = os.path.join(PROJECT_ROOT, "train_lstm.py")
if not os.path.exists(train_module_path):
    print("ERROR: train_lstm.py not found in project root. Place it here or rename accordingly.")
    sys.exit(1)

# Import train_lstm as a module (without running as script if it uses __main__)
spec = importlib.util.spec_from_file_location("train_lstm", train_module_path)
train_mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(train_mod)
    print("Imported train_lstm.py successfully.")
except Exception:
    print("Failed to import train_lstm.py — traceback follows:")
    traceback.print_exc()
    print("\nIf train_lstm.py runs training on import (instead of exposing a builder function), open it\nand find the function or code that builds the model (look for `model = Sequential(...)` or `def build_model`).")
    sys.exit(1)

# Try to locate a model builder or model instance in common places
candidate_builders = [
    "build_model", "create_model", "get_model", "make_model", "model_builder"
]
model_obj = None
builder_fn = None

# 1) If module defines `model` variable, try that first
if hasattr(train_mod, "model"):
    print("Found `model` in train_lstm.py — using that.")
    model_obj = getattr(train_mod, "model")

# 2) Try common builder function names
if model_obj is None:
    for name in candidate_builders:
        if hasattr(train_mod, name):
            print(f"Found builder function: {name}() — will call it (no args).")
            builder_fn = getattr(train_mod, name)
            try:
                model_obj = builder_fn()
                print("Builder returned a model instance.")
                break
            except TypeError as e:
                print(f"Calling {name}() raised TypeError (likely needs args): {e}")
                # we'll try calling with common args later

# 3) If builder function expects args, try common signatures
if model_obj is None and builder_fn is not None:
    # common guess values — tweak if your training used other values
    guesses = [
        {"timesteps":16, "features":63, "num_classes":2},
        {"TIMESTEPS":16, "FEATURES":63, "NUM_CLASSES":2},
        {"input_shape":(16,63), "num_classes":2},
        {"input_shape":(None,63), "num_classes":2},
        {}
    ]
    for g in guesses:
        try:
            print("Trying builder with args:", g)
            model_obj = builder_fn(**g)
            print("Builder returned a model with guessed args.")
            break
        except Exception as e:
            print("-> failed:", type(e).__name__, e)

# 4) If still no model, search for classes named Model or subclasses (rare)
if model_obj is None:
    # search attributes for a keras model class/instance
    for attr in dir(train_mod):
        val = getattr(train_mod, attr)
        # heuristic: if val has 'summary' attr, it's likely a model instance
        if hasattr(val, "summary") and hasattr(val, "save"):
            print(f"Found model-like object: {attr}")
            model_obj = val
            break

if model_obj is None:
    print("\nCould not automatically construct the model from train_lstm.py.\nPlease open train_lstm.py and copy here the code block that builds the model architecture (the part that defines `model = ...` or a `def build_model(...):` function).")
    sys.exit(1)

# At this point we have `model_obj` - a Keras Model instance without weights
print("Model summary (layer count):", len(model_obj.layers))
try:
    model_obj.summary()
except Exception as e:
    print("Could not call summary():", e)

# Ensure weights file exists
if not os.path.exists(WEIGHTS_PATH):
    print("ERROR: weights file not found at:", WEIGHTS_PATH)
    sys.exit(1)
print("Found weights file:", WEIGHTS_PATH, "size(bytes):", os.path.getsize(WEIGHTS_PATH))

# Now load weights
try:
    model_obj.load_weights(WEIGHTS_PATH)
    print("Weights loaded successfully into the model.")
except Exception:
    print("Failed to load weights with .load_weights(). Traceback:")
    traceback.print_exc()
    print("If load_weights failed due to name mismatches, confirm the layer names/ordering in train_lstm.py match those printed when you inspected the .keras file.")
    sys.exit(1)

# Finally save a full model as .h5
try:
    model_obj.save(OUT_H5)
    print("Saved full model to:", OUT_H5)
except Exception:
    print("Failed to save full model. Traceback:")
    traceback.print_exc()
    sys.exit(1)

print("Conversion complete — now update your GUI to point to:", OUT_H5)
