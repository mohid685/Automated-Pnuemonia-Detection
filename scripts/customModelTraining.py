import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from customModel import CustomDiseaseClassifier
from updatedPreprocessing import train_loader, val_loader
import time
from tqdm import tqdm

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
model = CustomDiseaseClassifier(num_classes=2).to(device)

# Hyperparameters
epochs = 50
learning_rate = 0.001

# Optimizer and Loss Function
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss()

# Logging with TensorBoard
writer = SummaryWriter("runs/custom_disease_classification")

# Training Loop
best_val_accuracy = 0.0
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    start_time = time.time()

    # Training Phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Training (Epoch {epoch+1})", leave=False)

    for batch_idx, (inputs, labels) in enumerate(train_loader_tqdm):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Compute average gradient norm
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item(), acc=100 * correct / total, grad_norm=grad_norm)

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation Phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    class_correct = [0] * 2
    class_total = [0] * 2
    val_loader_tqdm = tqdm(val_loader, desc=f"Validating (Epoch {epoch+1})", leave=False)

    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

            val_loader_tqdm.set_postfix(loss=loss.item(), acc=100 * val_correct / val_total)

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Per-Class Accuracy
    class_names = ['PNEUMONIA', 'NORMAL']
    print("Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{class_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    # Save Model for Each Epoch
    epoch_model_path = f"model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), epoch_model_path)
    print(f"\n Saved Model: {epoch_model_path}")

    # Save Best Model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "model7.pth")
        print(f"\n Saved Best Model (Epoch {epoch+1})")

    # Log Metrics
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
    writer.add_scalar('GradNorm/Train', grad_norm, epoch)

    print(f"Epoch Time: {time.time() - start_time:.2f}s")
    print(f"Best Validation Accuracy So Far: {best_val_accuracy:.2f}%")
