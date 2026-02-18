import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_TYPE = "MLP"   # change to "LSTM" if you want sequence model
MODEL_PATH = os.path.join("models", "final_landmark_model.h5") if MODEL_TYPE == "MLP" else os.path.join("models", "model_lstm.h5")
ENCODER_PATH_PKL = os.path.join("models", "label_encoder.pkl")

# ─────────────────────────────────────────────
# LOAD MODEL AND LABEL ENCODER
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
print(f"Loaded model: {MODEL_TYPE}")

# Load the label encoder from pickle only
if os.path.exists(ENCODER_PATH_PKL):
    with open(ENCODER_PATH_PKL, "rb") as f:
        le = pickle.load(f)
else:
    raise FileNotFoundError("Label encoder not found in models/ folder.")

print("Loaded label encoder, classes:", le.classes_)

# ─────────────────────────────────────────────
# MEDIAPIPE HAND DETECTION SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

current_output = ""   # to store the text sequence
last_pred = ""
frames_same = 0

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract 63 features (x,y,z for 21 landmarks)
            coords = []
            for p in handLms.landmark:
                coords += [p.x, p.y, p.z]
            X = np.array(coords).reshape(1, -1)

            # Predict using MLP
            pred = le.inverse_transform([np.argmax(model.predict(X))])[0]

            # Add prediction to text output only if stable
            if pred == last_pred:
                frames_same += 1
            else:
                frames_same = 0
            if frames_same > 10:  # about 0.4 sec stability
                current_output += pred
                frames_same = 0

            last_pred = pred

            # Draw prediction on frame
            cv2.putText(frame, f"Detected: {pred}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(frame, f"Text: {current_output}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Sign Language Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        current_output = ""   # clear text
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
