import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from customModel import CustomDiseaseClassifier
from updatedPreprocessing import test_loader

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded Model Path
BEST_MODEL_PATH = "C:/AI-PROJECT-25/trainedModels/model2.pth"  # Ensure the path is correct

# Load Model
model = CustomDiseaseClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

# Class Names
class_names = ['PNEUMONIA', 'NORMAL']


def evaluate_model(model, test_loader, class_names, device='cpu'):
    """Evaluate the model and display detailed metrics, including per-class accuracy."""
    all_labels = []
    all_predictions = []
    all_probs = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    # Overall Accuracy
    test_accuracy = 100 * test_correct / test_total
    print(f"\nOverall Test Accuracy: {test_accuracy:.2f}%")

    # Per-Class Accuracy Calculation
    print("\nPer-Class Accuracy:")
    cm = confusion_matrix(all_labels, all_predictions)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        print(f"{class_name} Accuracy: {per_class_accuracy[i] * 100:.2f}%")

    visualize_predictions(all_labels, all_predictions, all_probs, class_names, test_loader, device)


def visualize_predictions(true_labels, predicted_labels, probabilities, class_names, test_loader, device):
    """Display 30-40 test samples (15-20 from each class) with predictions."""
    fig, axes = plt.subplots(4, 10, figsize=(20, 10))
    axes = axes.flatten()

    pneumonia_count, normal_count = 0, 0
    max_samples_per_class = 20
    selected_images = []

    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        for i in range(len(images)):
            true_class = class_names[labels[i].item()]
            pred_class = class_names[preds[i].item()]
            prob = probs[i].max().item()

            if labels[i].item() == 0 and pneumonia_count < max_samples_per_class:
                selected_images.append((images[i], true_class, pred_class, prob))
                pneumonia_count += 1
            elif labels[i].item() == 1 and normal_count < max_samples_per_class:
                selected_images.append((images[i], true_class, pred_class, prob))
                normal_count += 1

            if pneumonia_count >= max_samples_per_class and normal_count >= max_samples_per_class:
                break
        if pneumonia_count >= max_samples_per_class and normal_count >= max_samples_per_class:
            break

    # Display Images
    for i in range(len(selected_images)):
        img, true_class, pred_class, prob = selected_images[i]
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class} ({prob:.2f})",
                          color='green' if true_class == pred_class else 'red', fontsize=8)
        axes[i].axis('off')

    # Adjust layout for better spacing and no overlap
    plt.tight_layout(pad=3.0)
    plt.show()


# Run Evaluation
evaluate_model(model, test_loader, class_names, device=device)

# best model:
# model2.pth: 85.58% accuracy