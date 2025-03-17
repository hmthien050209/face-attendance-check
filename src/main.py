# main.py - the main detection script

from datetime import datetime
import cv2

# from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")

# Export model to onnx
# model.export(format="onnx")

# load haarcascades model for face-detection task
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cv2.setUseOptimized(True)


def face_detection():
    last_date = datetime.now().second
    frame_count = 0
    # Open the default webcam
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
    if not cam.isOpened():
        print("Error: Cannot open the webcam")
        exit()

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Frame: %dx%d" % (frame_height, frame_width))

    while True:
        if datetime.now().second > last_date:
            last_date = datetime.now().second
            print("FPS: " + str(frame_count))
            frame_count = 0
        else:
            frame_count += 1

        ret, frame = cam.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=20, minSize=(128, 128)
        )

        # Draw rectangles around detected faces
        for x, y, w, h in faces:
            text = "face"
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Put text on bounding box
            cv2.putText(frame, text, (x, y - 5), font, 0.5, (255, 0, 0), 1)

        # Display the output
        cv2.imshow("Face Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


face_detection()
