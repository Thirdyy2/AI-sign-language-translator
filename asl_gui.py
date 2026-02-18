# asl_gui.py (updated)

import os
import threading
import time
import pickle
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
try:
    import joblib
except Exception:
    joblib = None

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

from tensorflow.keras.models import load_model

# ---------- Config ----------
DEBUG = True               # set False to suppress debug prints
STATIC_THRESHOLD = 0.60    # static model acceptance
SEQ_THRESHOLD_DEBUG = 0.55 # lowered threshold while debugging
SEQ_THRESHOLD_PROD = 0.75  # production threshold
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------- Helper: robust model loader ----------
def try_load_model(path):
    try:
        m = load_model(path)
        if DEBUG:
            print(f"Loaded model: {path} (input_shape={getattr(m,'input_shape',None)})")
        return m
    except Exception as e:
        if DEBUG:
            print(f"Failed to load model {path} -> {e}")
        return None

# Static model load
static_model = None
le_static = None
static_model_path = os.path.join(MODEL_DIR, "final_landmark_model.h5")
if os.path.exists(static_model_path):
    static_model = try_load_model(static_model_path)

# static encoder (.joblib preferred, fallback to pickle)
le_paths = [os.path.join(MODEL_DIR, "label_encoder.joblib"),
            os.path.join(MODEL_DIR, "label_encoder.pkl")]
for p in le_paths:
    if os.path.exists(p):
        try:
            if p.endswith(".joblib") and joblib:
                le_static = joblib.load(p)
            else:
                with open(p, "rb") as f:
                    le_static = pickle.load(f)
            if DEBUG:
                print("Loaded static label encoder:", p)
            break
        except Exception as e:
            print("Failed to load encoder", p, e)

# Sequence model load
seq_model = None
seq_paths = [os.path.join(MODEL_DIR, "final_seq_model.h5"),
             os.path.join(MODEL_DIR, r"C:\Users\Tanya\Desktop\sign_language_project\models\sign_language_lstm.keras"),
             os.path.join(MODEL_DIR, "final_seq_model.keras")]
for p in seq_paths:
    if os.path.exists(p):
        seq_model = try_load_model(p)
        if seq_model is not None:
            break

# sequence labels (numpy)
labels_seq = None
labels_seq_path = os.path.join(MODEL_DIR, "labels_seq.npy")
if os.path.exists(labels_seq_path):
    try:
        labels_seq = np.load(labels_seq_path, allow_pickle=True)
        if DEBUG:
            try:
                print("Loaded labels_seq:", labels_seq.shape, labels_seq[:10])
            except Exception:
                print("Loaded labels_seq, length:", len(labels_seq))
    except Exception as e:
        print("Failed to load labels_seq.npy:", e)

# Print statuses
print("Loaded models:", bool(static_model), bool(seq_model))
print("Encoders:", bool(le_static), False if labels_seq is None else (getattr(labels_seq,'size',0) > 0))

# Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

# GUI
class ASLApp:
    def __init__(self, root):
        self.root = root

        root.title("ASL Demo")
        self.vcap = cv2.VideoCapture(0)
        if not self.vcap.isOpened():
            messagebox.showerror("Camera error","Cannot open webcam.")
            root.destroy()
            return

        # Video label
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Output text
        self.output_var = tk.StringVar(value="")
        self.output_box = tk.Label(root, textvariable=self.output_var, bg="white",
                                   font=("Helvetica", 20), width=40, height=3, anchor="w", relief="sunken")
        self.output_box.pack(padx=8, pady=6, fill="x")

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=8, pady=6)
        tk.Button(btn_frame, text="Delete", command=self.delete_last, width=10).grid(row=0,column=0,padx=4)
        tk.Button(btn_frame, text="Clear", command=self.clear_all, width=10).grid(row=0,column=1,padx=4)

        # Predict-sequence toggle and status label
        self.seq_only = False  # when True: only use sequence model
        self.seq_button = tk.Button(btn_frame, text="Predict Sequence: OFF", command=self.toggle_seq_only, width=16 )
        self.seq_button.grid(row=0, column=2, padx=4)

        # Start/Stop prediction toggle button
        self.predict_mode = False  # when True: models predict, when False: skip model predictions
        self.predict_button = tk.Button(btn_frame, text="Start Prediction", command=self.toggle_prediction, width=14)
        self.predict_button.grid(row=0, column=3, padx=4)

        # Persistent status label (shows when seq-only is ON or prediction OFF)
        self.status_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"))
        self.status_label.pack(pady=(0,6))

        # smoothing
        self.smooth = 6  # how many frames stable to accept
        self.pred_buffer = []
        self.seq_buffer = []
        # TIMESTEPS infer
        if seq_model is not None:
            try:
                t = seq_model.input_shape[1]
                self.TIMESTEPS = 16 if t is None else int(t)
            except Exception:
                self.TIMESTEPS = 16
        else:
            self.TIMESTEPS = 16

        # internal
        self.sentence = ""
        self.last_app = None
        self.running = True
        # TTS not used any more (Speak removed) but keep initialization harmless
        self.tts = pyttsx3.init() if TTS_AVAILABLE else None

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

    def delete_last(self):
        self.sentence = self.sentence[:-1]
        self.output_var.set(self.sentence)

    def clear_all(self):
        self.sentence = ""
        self.output_var.set(self.sentence)

    def toggle_prediction(self):
        """Toggle global prediction ON/OFF."""
        self.predict_mode = not self.predict_mode
        if self.predict_mode:
            self.predict_button.config(text="Stop Prediction")
            # when starting, clear buffers and status
            self.pred_buffer.clear()
            self.seq_buffer.clear()
            self.last_app = None
            # update status (keep seq_only indicator too)
            self._update_status_label()
            if DEBUG: print("Predictions: STARTED")
        else:
            self.predict_button.config(text="Start Prediction")
            # when stopping, make sure no partial predictions leak
            self.pred_buffer.clear()
            self.seq_buffer.clear()
            self.last_app = None
            self._update_status_label()
            if DEBUG: print("Predictions: STOPPED")

    def toggle_seq_only(self):
        """Toggle sequence-only prediction mode ON/OFF."""
        self.seq_only = not self.seq_only
        # Update button label
        self.seq_button.config(text=f"Predict Sequence: {'ON' if self.seq_only else 'OFF'}")
        # Update persistent status label
        if self.seq_only:
            # clear static buffer so previous static predictions don't leak into seq-only usage
            self.pred_buffer.clear()
            self.last_app = None
        else:
            # clear seq buffer when turning off so old frames don't instantly trigger a sequence prediction
            self.seq_buffer.clear()
            self.last_app = None
        self._update_status_label()

    def _update_status_label(self):
        parts = []
        if not self.predict_mode:
            parts.append("PREDICTION: OFF")
        if self.seq_only:
            parts.append("SEQ ONLY MODE")
        self.status_label.config(text=" | ".join(parts), fg="red" if (not self.predict_mode or self.seq_only) else "black")

    def update_loop(self):
        if not self.running:
            return
        ret, frame = self.vcap.read()
        if not ret:
            self.root.after(30, self.update_loop); return
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        predicted = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # single-frame features
            coords = []
            for p in lm.landmark:
                coords += [p.x, p.y, p.z]
            coords = np.array(coords, dtype="float32").reshape(1,-1)

            # Only run model predictions if prediction mode is ON
            if self.predict_mode:
                # static prediction (skip when sequence-only mode is active)
                if (not self.seq_only) and static_model is not None and le_static is not None:
                    try:
                        probs = static_model.predict(coords, verbose=0)[0]
                        idx = int(np.argmax(probs))
                        label = le_static.inverse_transform([idx])[0] if hasattr(le_static,'inverse_transform') else None
                        # apply threshold
                        if probs.max() > STATIC_THRESHOLD:
                            self.pred_buffer.append(label)
                            if len(self.pred_buffer) > self.smooth: self.pred_buffer.pop(0)
                            # majority vote
                            if len(self.pred_buffer) >= self.smooth:
                                from collections import Counter
                                most = Counter(self.pred_buffer).most_common(1)[0][0]
                                predicted = most
                    except Exception as e:
                        if DEBUG: print("Static predict err:", e)

                # === sequence buffer (for motion) ===
                if seq_model is not None and labels_seq is not None:
                    try:
                        # convert coords (1,63) -> (21,3)
                        arr = np.array(coords).reshape(21,3)
                        # center by wrist
                        arr = arr - arr[0:1,:]

                        # simple scale normalization: divide by distance wrist -> middle_finger_mcp (index 9)
                        wrist = arr[0]
                        mid_mcp = arr[9]
                        scale = np.linalg.norm(mid_mcp - wrist)
                        if scale <= 1e-6:
                            scale = 1.0
                        arr = arr / scale

                        flat = arr.reshape(-1)
                        self.seq_buffer.append(flat)
                        if len(self.seq_buffer) > self.TIMESTEPS:
                            self.seq_buffer.pop(0)

                        # when buffer full, run seq model
                        if len(self.seq_buffer) == self.TIMESTEPS:
                            Xs = np.array(self.seq_buffer).reshape(1,self.TIMESTEPS, -1).astype("float32")
                            try:
                                probs_s = seq_model.predict(Xs, verbose=0)[0]
                                maxp = float(np.max(probs_s))
                                idx_s = int(np.argmax(probs_s))
                                seq_label = labels_seq[idx_s] if labels_seq is not None and idx_s < len(labels_seq) else str(idx_s)
                                if DEBUG:
                                    print(f"SEQ PRED -> label={seq_label} idx={idx_s} max_prob={maxp:.4f}")

                                # use a lower threshold for debugging, higher for production
                                thr = SEQ_THRESHOLD_DEBUG if DEBUG else SEQ_THRESHOLD_PROD

                                # if seq model confident and it differs (or seq-only mode is on), prefer seq
                                if maxp > thr:
                                    predicted = seq_label
                            except Exception as e:
                                if DEBUG: print("Seq predict err:", e)
                    except Exception as e:
                        if DEBUG: print("Seq preproc err:", e)
            else:
                # If predictions are disabled, keep buffers short to avoid lag when starting again
                if len(self.pred_buffer)>0: self.pred_buffer.pop(0)
                if len(self.seq_buffer)>0: self.seq_buffer.pop(0)

        else:
            # no hand
            if len(self.pred_buffer)>0: self.pred_buffer.pop(0)
            if len(self.seq_buffer)>0: self.seq_buffer.pop(0)

        # if predicted stable, append
        if predicted is not None:
            # simple stability check: append only if different from last appended
            if self.last_app != predicted:
                # handle special tokens
                if predicted == "SPACE":
                    self.sentence += " "
                elif predicted == "DEL":
                    self.sentence = self.sentence[:-1]
                else:
                    self.sentence += predicted
                self.output_var.set(self.sentence)
                self.last_app = predicted
        else:
            # If predictions are off, ensure last_app doesn't block new predictions when enabled
            if not self.predict_mode:
                self.last_app = None

        # show frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # update status label periodically
        self._update_status_label()

        self.root.after(20, self.update_loop)

    def on_close(self):
        self.running = False
        try: self.vcap.release()
        except: pass
        self.root.destroy()

#Final run for the functions.
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()


