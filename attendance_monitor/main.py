#!/usr/bin/env python3
import cv2, os

def main():
    name = input("Enter person's name (no spaces): ").strip()
    num_images = int(input("Number of images to capture (e.g. 20): ").strip())
    outdir = os.path.join("dataset", name)
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press SPACE to capture an image. Press ESC to quit early.")
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        cv2.putText(frame, f"{name}: {count}/{num_images}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capture (SPACE to save, ESC to exit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            path = os.path.join(outdir, f"{name}_{count+1}.jpg")
            cv2.imwrite(path, frame)
            print("Saved", path)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
