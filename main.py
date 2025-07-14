# import os
# from pathlib import Path
#
#
# ##### HELPER SCRIPT FOR COUNTING THE IMAGES WAS USED TO REDUCE THE BIAS FACTOR IN THE INTIAL PHASES #####
#
# def count_images_in_folder(folder_path, exclude_clahe=True):
#     """Count the number of image files in a folder and its subfolders, excluding _CLAHE folders if specified."""
#     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
#     count = 0
#
#     for root, dirs, files in os.walk(folder_path):
#         # Skip _CLAHE directories if exclude_clahe is True
#         if exclude_clahe:
#             dirs[:] = [d for d in dirs if '_CLAHE' not in d]
#
#         for file in files:
#             if Path(file).suffix.lower() in image_extensions:
#                 count += 1
#     return count
#
#
# # Define the paths to your folders
# base_path = "C:/AI-PROJECT-25/chest_xray"
# folders = ["test", "train", "val"]
#
# # Count images in each folder (excluding _CLAHE)
# print("Main folder counts (excluding _CLAHE folders):")
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     if os.path.exists(folder_path):
#         count = count_images_in_folder(folder_path, exclude_clahe=True)
#         print(f"Number of images in {folder}: {count}")
#     else:
#         print(f"Folder not found: {folder_path}")
#
# # Count images in subfolders (excluding _CLAHE)
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     if os.path.exists(folder_path):
#         print(f"\nBreakdown for {folder} (excluding _CLAHE folders):")
#         for subfolder in os.listdir(folder_path):
#             subfolder_path = os.path.join(folder_path, subfolder)
#             if os.path.isdir(subfolder_path) and '_CLAHE' not in subfolder:
#                 count = count_images_in_folder(subfolder_path, exclude_clahe=True)
#                 print(f"  {subfolder}: {count} images")