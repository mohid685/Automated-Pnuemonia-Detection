import traceback
import random
import torch
import torchvision.transforms as transforms
from fontTools.subset import subset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import io
import sys
import base64
import os
from PIL import Image

# Add path to import 'scripts.customModel'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.customModel import CustomDiseaseClassifier

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model Configuration
BEST_MODEL_PATH = "C:/AI-PROJECT-25/trainedModels/model2.pth"
class_names = ['PNEUMONIA', 'NORMAL']

# Test data path - VERIFY THIS MATCHES YOUR ACTUAL DATA STRUCTURE
TEST_DATA_PATH = "C:/AI-PROJECT-25/chest_xray/test"

# Image transformations
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
model = CustomDiseaseClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()


def create_balanced_test_loader(max_samples=40):
    """Create a balanced test loader with equal samples from both classes"""
    full_dataset = ImageFolder(
        root=TEST_DATA_PATH,
        transform=val_test_transforms
    )

    # Get indices for each class
    pneumonia_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]
    normal_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]

    # Sample equally from both classes
    samples_per_class = max_samples // 2
    pneumonia_samples = random.sample(pneumonia_indices, min(samples_per_class, len(pneumonia_indices)))
    normal_samples = random.sample(normal_indices, min(samples_per_class, len(normal_indices)))

    selected_indices = pneumonia_samples + normal_samples

    # Create subset and loader
    balanced_subset = torch.utils.data.Subset(full_dataset, selected_indices)
    return DataLoader(
        balanced_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

def tensor_to_base64(tensor):
    """Convert a tensor to base64 encoded JPEG"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def evaluate_model(test_loader, max_wrong_samples=3):
    """Batch evaluation with web-friendly output"""
    all_labels = []
    all_preds = []
    sample_results = []
    wrong_samples_collected = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for i in range(inputs.size(0)):
                is_correct = preds[i].item() == labels[i].item()

                # Always include correct predictions
                if is_correct:
                    sample_results.append({
                        'image': tensor_to_base64(inputs[i]),
                        'true_label': class_names[labels[i].item()],
                        'prediction': class_names[preds[i].item()],
                        'confidence': round(probs[i][preds[i].item()].item() * 100, 2),
                        'correct': True
                    })
                # Include only a few wrong predictions
                elif wrong_samples_collected < max_wrong_samples:
                    sample_results.append({
                        'image': tensor_to_base64(inputs[i]),
                        'true_label': class_names[labels[i].item()],
                        'prediction': class_names[preds[i].item()],
                        'confidence': round(probs[i][preds[i].item()].item() * 100, 2),
                        'correct': False
                    })
                    wrong_samples_collected += 1

    accuracy = 100 * np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)

    return {
        'overall_accuracy': round(accuracy, 2),
        'samples': sample_results,
        'total_samples': len(all_labels)
    }

def run_batch_evaluation():
    """Main function for batch processing"""
    try:
        try:
            from updatedPreprocessing import test_loader
            print("Using test_loader from updatedPreprocessing", file=sys.stderr)
        except ImportError:
            print("Using balanced test loader", file=sys.stderr)
            test_loader = create_balanced_test_loader()

        results = evaluate_model(test_loader)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({
            'error': str(e),
            'traceback': traceback.format_exc()
        })


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        print(run_batch_evaluation())
    else:
        if len(sys.argv) < 2:
            print(json.dumps({
                'error': 'No image path provided',
                'usage': 'python predict.py [--batch] or [image_path]'
            }))
            sys.exit(1)

        try:
            image = Image.open(sys.argv[1]).convert('RGB')
            image = val_test_transforms(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

            result = {
                'prediction': class_names[pred.item()],
                'confidence': round(probs[0][pred.item()].item() * 100, 2),
                'image': tensor_to_base64(image.squeeze(0))
            }
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({
                'error': str(e),
                'traceback': traceback.format_exc()
            }))