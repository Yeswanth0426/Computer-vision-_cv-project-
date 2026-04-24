import cv2
import os
import pickle
import numpy as np

dataset_path = "dataset"

encodings = []
names = []

def get_embedding(img):
    img = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize brightness
    gray = cv2.equalizeHist(gray)

    return gray.flatten()

print("[INFO] Encoding faces...")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        emb = get_embedding(image)

        encodings.append(emb)
        names.append(person)

print("[INFO] Total encodings:", len(names))

data = {"encodings": encodings, "names": names}

with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("✅ Encoding complete")
