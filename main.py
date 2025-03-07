import cv2 
import numpy
import torch
import os
from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11n.pt")

# Export model to onnx
#model.export(format="onnx")

# load haarcascades model for face-detection task
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

def face_detection():
    # Open the default webcam
    cam = cv2.VideoCapture(0)
        
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
#        out.write(frame)

          # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #cv2.putText('Face', (x, y-10), 
            #      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the output
        cv2.imshow('Face Detection', frame)

            # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
#    out.release()
    cv2.destroyAllWindows()
face_detection()

