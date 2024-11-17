import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm  # Used for displaying progress bars

# Read the label file
df = pd.read_csv("dataset/dataset_labels.csv")

# Define the folder path for storing processed images
processed_data_path = "dataset/processed_dataset"
os.makedirs(processed_data_path, exist_ok=True)

# Define the CSV file path for the processed dataset
processed_csv_path = "dataset/processed_dataset_labels.csv"

# Resize the image
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

# Convert to HSV color space
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Image augmentation (rotation and horizontal flip)
def augment_image(image):
    # Random rotation
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Random horizontal flip
    if np.random.rand() > 0.5:
        rotated = cv2.flip(rotated, 1)
        
    return rotated

# Complete image preprocessing pipeline
def preprocess_image(image):
    image = resize_image(image, size=(256, 256))
    image = convert_to_hsv(image)
    image = adjust_brightness_contrast(image)
    return image

# Function to process and save images
def process_and_save_images():
    processed_data = []  # Used to save information about processed data

    # Use tqdm to add a progress bar for the dataset
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = row['image_path']
        state = row['state']
        fruit_type = row['fruit_type']
        
        # Check if the image path exists
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            continue
        
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            continue

        # Image preprocessing steps
        processed_image = preprocess_image(image)

        # Create the corresponding subfolder
        processed_folder = os.path.join(processed_data_path, f"{state}_{fruit_type}")
        os.makedirs(processed_folder, exist_ok=True)

        # Construct the save path for the processed image
        processed_image_path = os.path.join(processed_folder, os.path.basename(image_path))
        
        # Save the preprocessed image
        cv2.imwrite(processed_image_path, processed_image)

        # Record the original labels and processed image path in the dataset
        processed_data.append({
            "processed_image_path": processed_image_path,
            "state": state,
            "fruit_type": fruit_type
        })

        # Generate multiple augmentations for each image
        for aug_index in range(3):  # Generate 3 augmented versions for each image
            augmented_image = augment_image(processed_image)
            
            # Construct the save path for the augmented image
            augmented_image_path = os.path.join(
                processed_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{aug_index}.jpg"
            )
            
            # Save the augmented image
            cv2.imwrite(augmented_image_path, augmented_image)

            # Record the augmented image path and labels in the dataset
            processed_data.append({
                "processed_image_path": augmented_image_path,
                "state": state,
                "fruit_type": fruit_type
            })

    # Save the processed data paths to a CSV file
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(processed_csv_path, index=False)
    print(f"Processed image paths have been saved to {processed_csv_path}")

# Run the image processing and saving function
process_and_save_images()
