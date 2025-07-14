import os
from glob import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image

# Custom dataset loader for raw images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label=None):
        self.image_paths = glob(os.path.join(image_dir, "*.jpeg")) + \
                           glob(os.path.join(image_dir, "*.png")) + \
                           glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform
        self.label = label  # 0 for NORMAL, 1 for PNEUMONIA

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label

# Paths to data directories
train_dir = r"C:/AI-PROJECT-25/chest_xray/train"
pneumonia_dir = r"C:/AI-PROJECT-25/chest_xray/train/PNEUMONIA"
normal_dir = r"C:/AI-PROJECT-25/chest_xray/train/NORMAL"  # Use original NORMAL directory
val_dir = r"C:/AI-PROJECT-25/chest_xray/val"
test_dir = r"C:/AI-PROJECT-25/chest_xray/test"

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load NORMAL images without CLAHE (Label = 0)
normal_dataset = CustomImageDataset(image_dir=normal_dir, transform=train_transforms, label=0)

# Load CLAHE-processed PNEUMONIA images (Label = 1)
pneumonia_dataset = CustomImageDataset(image_dir=pneumonia_dir, transform=train_transforms, label=1)

# Combine both datasets
train_dataset = ConcatDataset([normal_dataset, pneumonia_dataset])

# Load validation and test datasets
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset sizes
print(f"Training data: {len(train_dataset)} images")
print(f"Validation data: {len(val_dataset)} images")
print(f"Test data: {len(test_dataset)} images")
