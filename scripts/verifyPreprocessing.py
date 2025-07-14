# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from updatedPreprocessing import train_loader  # Import train_loader from preprocess.py

# Define a function to denormalize images
def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

# Use train_loader to visualize a batch of images
data_iter = iter(train_loader)
images, labels = next(data_iter)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Plot images to verify preprocessing
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    ax = axes[i]
    image = denormalize(images[i], mean, std)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"Label: {labels[i].item()}")
plt.show()

print(f"Image batch shape: {images.shape}")
