# main.py - the main detection script

from ultralytics import YOLO
import cv2

print("Starting the detection process...")

print("Loading the trained model...")
# Load pretrained model in the 'trained.pt' file
model = YOLO("trained.pt")

print("Detecting from the webcam...")

cap = cv2.VideoCapture(0)
cv2.setUseOptimized(True)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.predict(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("People detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
