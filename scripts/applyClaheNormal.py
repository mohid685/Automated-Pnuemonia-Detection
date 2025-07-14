import cv2
import os
from glob import glob
from tqdm import tqdm

# Input and output directories
input_dir = r"C:/AI-PROJECT-25/chest_xray/train/NORMAL"
output_dir = r"C:/AI-PROJECT-25/chest_xray/train/NORMAL_CLAHE"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image
for img_path in tqdm(glob(os.path.join(input_dir, "*.jpeg")) +
                      glob(os.path.join(input_dir, "*.png")) +
                      glob(os.path.join(input_dir, "*.jpg"))):

    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Save processed image
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, img_clahe)
