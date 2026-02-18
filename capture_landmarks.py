# capture_landmarks.py
import cv2, os, csv, time
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

label = input("Enter label (A-Z): ").strip().upper()
out_dir = "landmark_data"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f"{label}.csv")

print("Controls: press 's' to start/pause capturing, 'q' to quit.")

cap = cv2.VideoCapture(0)
capturing = False
count = 0

# create CSV with header if not exists
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header += ["label"]
        writer.writerow(header)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available")
        break
    frame = cv2.flip(frame, 1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        coords = []
        for p in lm.landmark:
            coords += [p.x, p.y, p.z]
        cv2.putText(frame, f"Hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if capturing:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(coords + [label])
            count += 1
            cv2.putText(frame, f"Captured: {count}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Capture Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        capturing = not capturing
        print("Capturing:", capturing)
    elif key == ord('q'):
        break

cap.release()
cv2.waitkey(0)
cv2.destroyAllWindows()
