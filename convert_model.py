# convert_model.py
import os
from tensorflow.keras.models import load_model

model_path = r"C:\Users\Tanya\Desktop\sign_language_project\models\sign_language_lstm.keras"
save_path  = r"C:\Users\Tanya\Desktop\sign_language_project\models\sign_language_lstm.h5"

print("Checking that file exists:", model_path)
if not os.path.exists(model_path):
    raise SystemExit("ERROR: model file not found at: " + model_path)

print("Loading model (this requires a TF/Keras build that supports .keras files)...")
m = load_model(model_path)

print("Saving to .h5:", save_path)
m.save(save_path)
print("Done — saved:", save_path)
