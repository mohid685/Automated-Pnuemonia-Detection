import torch
from torch import nn

# Custom CNN Model
class CustomDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomDiseaseClassifier, self).__init__()

        # Feature Extraction Layers (Conv + ReLU + Pooling)d
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downscale by 2x

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downscale by 2x

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downscale by 2x

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Downscale by 2x
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten feature maps
            nn.Linear(256 * 14 * 14, 512),  # Adjust size if needed
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Instantiate Model
model = CustomDiseaseClassifier(num_classes=2).to('cuda' if torch.cuda.is_available() else 'cpu')
