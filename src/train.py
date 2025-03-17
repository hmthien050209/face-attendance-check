# train.py - The main training script

# Imports
import glob
import math
import os
from random import shuffle
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Constants
SEED = 37573


# Define the Training class to handle the training process
class Training:
    def __init__(self, path_to_dataset: str, imgsz: int):
        self.path_to_dataset = path_to_dataset
        self.imgsz = imgsz

    def split_images(self):
        input_folder = self.path_to_dataset
        output_folder = Path(self.path_to_dataset).parent / "splitted_dataset"
        self.path_to_dataset = output_folder
        # If the folder already exists, remove it
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Create the output folder
        os.makedirs(output_folder, exist_ok=True)

        # Define the categories and split ratios for training, testing, and validation sets
        categories = [
            os.path.basename(p)
            for p in glob.glob(os.path.join(input_folder, "*"))
            if os.path.isdir(p)
        ]
        print(categories)
        split_ratios = {"train": 70, "test": 20, "val": 10}  # Total parts = 100

        for category in categories:
            # List all PNG images in the current category folder from the input folder
            image_paths = glob.glob(os.path.join(input_folder, category, "*.jpg"))
            if not image_paths:
                print(f"No images found in {os.path.join(input_folder, category)}.")
                continue

            # Shuffle images to ensure random splitting
            shuffle(image_paths)
            total = len(image_paths)

            # Compute split counts
            train_count = math.floor(total * split_ratios["train"] / 100)
            test_count = math.floor(total * split_ratios["test"] / 100)
            val_count = total - train_count - test_count  # Use remaining images for val

            # Split the image list
            train_images = image_paths[:train_count]
            test_images = image_paths[train_count : train_count + test_count]
            val_images = image_paths[train_count + test_count :]

            # Create output directories for each subset & category
            for subset in ["train", "test", "val"]:
                out_dir = os.path.join(output_folder, subset, category)
                os.makedirs(out_dir, exist_ok=True)

            # Copy the images
            for img_path in train_images:
                output_path = os.path.join(
                    output_folder, "train", category, os.path.basename(img_path)
                )
                shutil.copy(img_path, output_path)
            print(
                f"Category '{category}': Copied {len(train_images)} images for train set."
            )

            for img_path in test_images:
                output_path = os.path.join(
                    output_folder, "test", category, os.path.basename(img_path)
                )
                shutil.copy(img_path, output_path)
            print(
                f"Category '{category}': Copied {len(test_images)} images for test set."
            )

            for img_path in val_images:
                output_path = os.path.join(
                    output_folder, "val", category, os.path.basename(img_path)
                )
                shutil.copy(img_path, output_path)
            print(
                f"Category '{category}': Copied {len(val_images)} images for val set."
            )

    def start_training(self):
        print("Starting training")
        # Load the pretrained yolo11l-cls model
        model = YOLO("yolo11l-cls.pt")
        # Train the model
        results = model.train(
            data=self.path_to_dataset,
            epochs=60,
            batch=32,
            imgsz=self.imgsz,
            dropout=0.1,
            degrees=180,
            fliplr=0.5,
            flipud=0.5,
            multi_scale=True,
            erasing=0.0,
            seed=SEED,
        )
        # Save the trained model
        model.save("trained_model.pt")
        return results


# The main program
def main(path_to_dataset: str, imgsz: int):
    training = Training(path_to_dataset, imgsz)
    training.split_images()  # Split images into train, test, and val sets
    results = training.start_training()
    print(f"Training completed with validation accuracy: {results.val()}")


if __name__ == "__main__":
    # Run the main program if this script is executed directly
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_dataset> <imgsz>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
