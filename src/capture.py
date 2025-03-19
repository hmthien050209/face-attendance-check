# capture.py - capturing user's faces
import os
import shutil
from time import sleep

import cv2

# Constants
DELAY = 30 / 1000  # 30 FPS

# Face limits
FACE_NUM = 50
FACE_MIN_NEIGHBORS = 15
FACE_MIN_WIDTH = 128
FACE_MIN_HEIGHT = 128
FACE_PER_STYLE_COUNT = 10

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Load YOLO pretrained model

cv2.setUseOptimized(True)


def capture_faces(output_dir: str):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))

    if not cam.isOpened():
        print("Error: Cannot open the webcam")
        exit()

    count = 0
    idx = 0
    # Create the guide for users
    guide = [
        "look straight",
        "look slightly right",
        "look slightly left",
        "look slightly down",
        "look slightly up",
        "done",
    ]
    print("Started capturing")
    print(guide[idx])
    print("Press any key to continue...")
    while count < FACE_NUM:
        ret, frame = cam.read()
        # Boost brightness if necessary
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=50)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(FACE_MIN_WIDTH, FACE_MIN_HEIGHT),
        )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            count += 1

            # Resize the image to 320x320 pixels
            resized_frame = cv2.resize(frame[y : y + h, x : x + w], (320, 320))

            # Save the face image
            cv2.imwrite(f"{output_dir}/face_{count}.jpg", resized_frame)

            print(f"Captured {count} faces")

            if count % FACE_PER_STYLE_COUNT == 0:
                if idx < len(guide):
                    idx += 1
                    print(guide[idx])
                print("Press any key to continue...")
                # wait for user input to continue to next style guide
                input()

        cv2.imshow("Face capture", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("c) Capture user's face")
        print("q) Quit")
        cmd = input("Enter the command: ")
        if cmd == "c":
            name = input("Name: ")
            output_dir = f"captured/{name}/"
            os.makedirs(output_dir, exist_ok=True)
            capture_faces(output_dir)
        elif cmd == "q":
            shutil.make_archive("captured", "zip", "captured")
            print("Captured faces are saved in a zip file named 'captured.zip'")
            exit(0)


if __name__ == "__main__":
    main()
