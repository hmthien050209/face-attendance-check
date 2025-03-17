# capture.py - capturing user's faces
from time import sleep
import cv2
import os

# Constants
DELAY = 5 / 1000  # 5 FPS

# Face limits
FACE_NUM = 40
FACE_MIN_NEIGHBORS = 20
FACE_MIN_WIDTH = 128
FACE_MIN_HEIGHT = 128

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cv2.setUseOptimized(True)


def capture_faces(output_dir: str):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))

    if not cam.isOpened():
        print("Error: Cannot open the webcam")
        exit()

    count = 0
    print("Started capturing")
    while count < FACE_NUM:
        ret, frame = cam.read()
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
            frame = cv2.resize(frame[y : y + h, x : x + w], (320, 320))

            # Save the face image
            cv2.imwrite(f"{output_dir}/face_{count}.jpg", frame)

            cv2.imshow("Face capture", frame)

            print(f"Captured {count} faces")

        sleep(DELAY)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


def main():
    num = int(input("Enter the number of people to capture: "))
    for i in range(num):
        name = input("Name: ")
        output_dir = f"captured/{name}/"
        os.makedirs(output_dir, exist_ok=True)
        capture_faces(output_dir)


if __name__ == "__main__":
    main()
