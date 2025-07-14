import traceback

import torch
import torchvision.transforms as transforms
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
from customModel import CustomDiseaseClassifier

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration
BEST_MODEL_PATH = "C:/AI-PROJECT-25/trainedModels/model2.pth"
class_names = ['PNEUMONIA', 'NORMAL']

# Default test data path (modify as needed)
TEST_DATA_PATH = "C:/AI-PROJECT-25/data/test"

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


def create_test_loader():
    """Create a test loader if updatedPreprocessing is not available"""
    test_dataset = ImageFolder(
        root=TEST_DATA_PATH,
        transform=val_test_transforms
    )
    return DataLoader(
        test_dataset,
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


def evaluate_model(test_loader, max_samples=40):
    """Batch evaluation with web-friendly output"""
    all_labels = []
    all_preds = []
    sample_results = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if len(sample_results) >= max_samples:
                break

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for i in range(inputs.size(0)):
                if len(sample_results) >= max_samples:
                    break

                sample_results.append({
                    'image': tensor_to_base64(inputs[i]),
                    'true_label': class_names[labels[i].item()],
                    'prediction': class_names[preds[i].item()],
                    'confidence': round(probs[i][preds[i].item()].item() * 100, 2),
                    'correct': preds[i].item() == labels[i].item()
                })

    accuracy = 100 * np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    class_acc = {class_names[i]: round(cm[i, i] / cm[i].sum() * 100, 2) for i in range(len(class_names))}

    return {
        'overall_accuracy': round(accuracy, 2),
        'class_accuracies': class_acc,
        'confusion_matrix': cm.tolist(),
        'samples': sample_results,
        'total_samples': len(all_labels)
    }


def run_batch_evaluation():
    """Main function for batch processing"""
    try:
        try:
            from updatedPreprocessing import test_loader
        except ImportError:
            print("Note: updatedPreprocessing not found, using default test loader", file=sys.stderr)
            test_loader = create_test_loader()

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