import cv2
import os

DATASET_DIR = "dataset"

print("🔧 Fixing dataset images...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"Processing: {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            img = cv2.imread(img_path)

            if img is None:
                print("❌ Cannot read:", img_path)
                continue

            # FORCE convert to 8-bit 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # overwrite same file as clean JPG
            cv2.imwrite(img_path, img)

            print("✅ Fixed:", img_name)

        except Exception as e:
            print("Error:", img_name, e)

print("🎉 Dataset fixed!")
