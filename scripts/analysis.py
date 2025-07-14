import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from customModel import CustomDiseaseClassifier

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = "C:/AI-PROJECT-25/trainedModels/model2.pth"
model = CustomDiseaseClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Class Names
class_names = ['PNEUMONIA', 'NORMAL']
class_to_idx = {'PNEUMONIA': 1, 'NORMAL': 0}

# Transform (use same as training/validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        pred_label = class_names[pred_idx]
        prob = probabilities[0][pred_idx].item()

    return pred_label, prob


def evaluate_directory(directory_path):
    correct = 0
    total = 0

    print(f"\nEvaluating images in: {directory_path}\n")

    for class_folder in ['NORMAL', 'PNEUMONIA']:
        class_path = os.path.join(directory_path, class_folder)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist.")
            continue

        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                true_label = class_folder
                predicted_label, prob = predict_image(img_path)

                is_correct = predicted_label == true_label
                total += 1
                correct += 1 if is_correct else 0

                print(f"[{img_file}] True: {true_label:<9} | Predicted: {predicted_label:<9} | "
                      f"Confidence: {prob:.2f} | {'✔️' if is_correct else '❌'}")

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n✅ Accuracy: {accuracy:.2f}% on {total} images.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_folder.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    evaluate_directory(directory)
