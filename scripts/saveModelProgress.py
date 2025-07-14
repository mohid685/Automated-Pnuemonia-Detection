import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import classification_report
from customModel import CustomDiseaseClassifier
from updatedPreprocessing import test_loader

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Names
class_names = ['PNEUMONIA', 'NORMAL']

# Training parameters from customModelTraining.py
TRAINING_PARAMS = {
    "model_architecture": "CustomDiseaseClassifier",
    "num_classes": 2,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "batch_size": 32,
    "image_size": 224
}


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model and return basic metrics."""
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    # Calculate overall accuracy
    test_accuracy = 100 * test_correct / test_total

    return {
        "overall_accuracy": round(test_accuracy, 2),
        "total_samples": int(test_total)
    }


def get_model_performance(model_path, device):
    """Load and evaluate a single model."""
    model = CustomDiseaseClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return evaluate_model(model, test_loader, device)


def save_model_records(records):
    """Save model records to a JSON file in ProgressReports directory."""
    # Create ProgressReports directory if it doesn't exist
    reports_dir = "ProgressReports"
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(reports_dir, f"model_performance_report_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(records, f, indent=4)

    print(f"Model performance records saved to {filename}")


def main():
    # Find all .pth files in trainedModels directory
    models_dir = "C:/AI-PROJECT-25/trainedModels"
    if not os.path.exists(models_dir):
        print(f"Directory '{models_dir}' not found.")
        return

    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])
    model_paths = [os.path.join(models_dir, f) for f in model_files]

    if not model_files:
        print(f"No .pth model files found in {models_dir} directory.")
        return

    print(f"Found {len(model_files)} model files to evaluate in {models_dir}")

    records = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "device_used": str(device),
            "class_names": class_names,
            "models_directory": models_dir
        },
        "training_parameters": TRAINING_PARAMS,
        "model_performance": {}
    }

    for model_file, model_path in zip(model_files, model_paths):
        print(f"Evaluating {model_file}...")
        try:
            metrics = get_model_performance(model_path, device)
            records["model_performance"][model_file] = metrics
            print(f"  Accuracy: {metrics['overall_accuracy']:.2f}%")
        except Exception as e:
            print(f"  Error evaluating {model_file}: {str(e)}")
            records["model_performance"][model_file] = {
                "error": str(e),
                "status": "evaluation_failed"
            }

    # Save the records
    save_model_records(records)

    # Generate performance summary
    print("\nPerformance Summary:")
    valid_models = {
        k: v for k, v in records["model_performance"].items()
        if "overall_accuracy" in v
    }

    if valid_models:
        print("{:<25} {:<10}".format("Model", "Accuracy"))
        print("-" * 35)

        for model_name, metrics in valid_models.items():
            print("{:<25} {:<10.2f}%".format(
                model_name,
                metrics["overall_accuracy"]))

        best_model = max(valid_models.items(),
                         key=lambda x: x[1]["overall_accuracy"])
        print(f"\nBest performing model: {best_model[0]} "
              f"(Accuracy: {best_model[1]['overall_accuracy']:.2f}%)")


if __name__ == "__main__":
    main()