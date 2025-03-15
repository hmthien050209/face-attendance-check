# train.py - The main training script

# Imports
from ultralytics import YOLO
import sys


# Define the Training class to handle the training process
class Training:
    def __init__(self, path_to_dataset: str, imgsz: int):
        self.path_to_dataset = path_to_dataset
        self.imgsz = imgsz

    def start_training(self):
        # Load the pretrained yolo11l-cls model
        model = YOLO("yolo11l-cls.pt")
        # Train the model
        results = model.train(
            data=self.path_to_dataset, epochs=60, batch=64, imgsz=self.imgsz
        )
        # Save the trained model
        model.save("trained_model.pt")
        return results


# The main program
def main(path_to_dataset: str):
    training = Training(path_to_dataset)
    results = training.start_training()
    print(f"Training completed with validation accuracy: {results.val()}")


if __name__ == "__main__":
    # Run the main program if this script is executed directly
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_dataset> <imgsz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
