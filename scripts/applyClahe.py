import os
import cv2
import numpy as np
from tqdm import tqdm

# Define input and output directories
input_dir = r"C:/AI-PROJECT-25/chest_xray/train/PNEUMONIA"
output_dir = r"C:/AI-PROJECT-25/chest_xray/train/PNEUMONIA_CLAHE"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# CLAHE parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Process all images
for img_name in tqdm(os.listdir(input_dir), desc="Applying CLAHE"):
    img_path = os.path.join(input_dir, img_name)
    output_path = os.path.join(output_dir, img_name)

    # Read the image in grayscale (since X-ray images are usually single-channel)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # Apply CLAHE
        img_clahe = clahe.apply(img)

        # Save the processed image
        cv2.imwrite(output_path, img_clahe)
    else:
        print(f"Skipping {img_name}, unable to read file.")

print("✅ CLAHE processing completed successfully.")
