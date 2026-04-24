import os
import cv2
import numpy as np
import face_recognition
import requests
from datetime import datetime

# ================== TELEGRAM ==================
BOT_TOKEN = "8378557687:AAFTAjHdcnYUDYKLiryNtbIVYlYd5cc_Hhc"
CHAT_ID = "6338488112"

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": msg}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        print("Telegram Error:", e)

# ================== SETTINGS ==================
DATASET_PATH = "dataset"
MATCH_THRESHOLD = 0.6   # IMPORTANT
FRAME_SCALE = 0.5

# ================== LOAD DATASET ==================
known_encodings = []
known_names = []

print("[INFO] Loading dataset...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    print(f"  -> {person}")

    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)

        try:
            image = cv2.imread(img_path)

            if image is None:
                print("❌ Skipped:", img_path)
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person)
            else:
                print("⚠️ No face found:", img_path)

        except Exception as e:
            print("Error:", img_path, e)

print("[INFO] Total faces loaded:", len(known_names))

if len(known_names) == 0:
    print("❌ No valid faces found. Fix dataset first.")
    exit()

# ================== START CAMERA ==================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

print("[INFO] Camera started... Press ESC to exit")

last_sent_time = 0
cooldown = 5  # seconds

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb_small, model="hog")
    encodings = face_recognition.face_encodings(rgb_small, faces)

    for (top, right, bottom, left), face_enc in zip(faces, encodings):

        name = "Unknown"

        distances = face_recognition.face_distance(known_encodings, face_enc)

        if len(distances) > 0:
            min_dist = np.min(distances)
            index = np.argmin(distances)

            if min_dist < MATCH_THRESHOLD:
                name = known_names[index]

        # Scale back to original size
        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)

        # Draw box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ================= TELEGRAM ALERT =================
        current_time = time_now = datetime.now().timestamp()

        if current_time - last_sent_time > cooldown:
            if name == "Unknown":
                send_telegram("⚠️ Unknown person detected!")
            else:
                send_telegram(f"✅ {name} detected")

            last_sent_time = current_time

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
