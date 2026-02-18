# infer_sequence.py (fixed: flexible paths for model + labels)
import os, numpy as np, cv2, mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

MODEL_DIR = "models"

# find model file (try a few common names)
possible_models = [
    os.path.join(MODEL_DIR, "final_seq_model.h5"),
    os.path.join(MODEL_DIR, "final_seq_model.keras"),
    os.path.join(MODEL_DIR, "sign_language_lstm.keras"),
    os.path.join(MODEL_DIR, "sign_language_lstm.h5"),
    os.path.join(MODEL_DIR, "models.h5"),
]
model_path = None
for p in possible_models:
    if os.path.exists(p):
        model_path = p
        break

# if not found, try any .h5 or .keras in models/
if model_path is None:
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith((".h5", ".keras")):
            model_path = os.path.join(MODEL_DIR, f)
            break

if model_path is None:
    raise FileNotFoundError("No LSTM model file found in 'models/'. Put your trained LSTM model (e.g. final_seq_model.h5) into models/ or update the script.")

print("Loading model:", model_path)
model = load_model(model_path)

# load labels: prefer label_encoder_seq.pkl/joblib, else fallback to labels_seq.npy
labels = None
pkl = os.path.join(MODEL_DIR, "label_encoder_seq.pkl")
job = os.path.join(MODEL_DIR, "label_encoder_seq.joblib")
npy = os.path.join(MODEL_DIR, "labels_seq.npy")

if os.path.exists(job):
    import joblib
    le = joblib.load(job)
    labels = list(le.classes_)
elif os.path.exists(pkl):
    import pickle
    with open(pkl, "rb") as f:
        le = pickle.load(f)
    labels = list(le.classes_)
elif os.path.exists(npy):
    labels = list(np.load(npy))
else:
    raise FileNotFoundError("No label encoder found. Expected label_encoder_seq.pkl/joblib or labels_seq.npy in models/")

print("Classes:", labels)

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

TIMESTEPS = model.input_shape[1]  # e.g. 30
buffer = deque(maxlen=TIMESTEPS)
pred_buffer = deque(maxlen=7)

cap = cv2.VideoCapture(0)
last_spoken = None
sentence = ""

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        coords = []
        for p in lm.landmark:
            coords += [p.x, p.y, p.z]
        arr = np.array(coords).reshape(21,3)
        arr = arr - arr[0:1,:]   # center by wrist
        flat = arr.reshape(-1)
        buffer.append(flat)

        if len(buffer) == TIMESTEPS:
            X = np.array(buffer).reshape(1, TIMESTEPS, flat.size).astype('float32')
            probs = model.predict(X, verbose=0)[0]
            idx = int(probs.argmax())
            label = labels[idx]
            pred_buffer.append(label)
            # smoothing majority vote
            most = Counter(pred_buffer).most_common(1)[0][0]
            cv2.putText(frame, f"Pred: {most}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # stable rule: require n votes
            if pred_buffer.count(most) >= 4 and (len(sentence)==0 or sentence[-1] != most):
                sentence += most
    else:
        # slowly clear buffers when no hand
        if len(pred_buffer)>0:
            pred_buffer.popleft()

    cv2.putText(frame, sentence, (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imshow("Seq Infer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
