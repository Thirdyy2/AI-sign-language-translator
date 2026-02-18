# capture_sequence.py
import cv2, os, time, numpy as np
import mediapipe as mp

# CONFIG
TIMESTEPS = 30                 # frames per sample, change if you want 12-20
OUTPUT_ROOT = "seq_data"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

label = input("Enter label for sequence capture (e.g. J or Z): ").strip().upper()
out_dir = os.path.join(OUTPUT_ROOT, label)
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Instructions:")
print(" - Press 's' to capture one sequence of", TIMESTEPS, "frames.")
print(" - Press 'q' to quit.")
print("Tip: For letter J start from 'I' pose then draw J with the pinky. Keep steady at end.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break
    frame = cv2.flip(frame, 1)
    disp = frame.copy()
    cv2.putText(disp, f"Label: {label}  Press 's' to capture sequence, 'q' to quit", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Sequence Capture", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        seq = []
        print("Capturing sequence...")
        for i in range(TIMESTEPS):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                coords = []
                for p in lm.landmark:
                    coords += [p.x, p.y, p.z]
                seq.append(coords)
            else:
                # if hand lost, append previous or zeros
                if len(seq) > 0:
                    seq.append(seq[-1])
                else:
                    seq.append([0.0]*63)
            # optional feedback
            temp = frame.copy()
            if res.multi_hand_landmarks:
                mp_draw.draw_landmarks(temp, lm, mp_hands.HAND_CONNECTIONS)
            cv2.putText(temp, f"Frame {i+1}/{TIMESTEPS}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Sequence Capture", temp)
            cv2.waitKey(int(1000/25))   # approx 25 fps
        seq = np.array(seq)   # shape (TIMESTEPS, 63)
        # center by wrist per frame (landmark 0)
        seq3 = seq.reshape(TIMESTEPS, 21, 3)
        wrist = seq3[:, 0:1, :]           # (TIMESTEPS,1,3)
        seq_center = seq3 - wrist         # broadcast subtract
        seq_flat = seq_center.reshape(TIMESTEPS, 63)
        fname = f"{label}_{int(time.time()*1000)}.npy"
        np.save(os.path.join(out_dir, fname), seq_flat)
        print("Saved sequence:", os.path.join(out_dir, fname))
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
