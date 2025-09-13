import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import json

# Import your custom modules
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'dog_and_cat_classifier_cnn_from_scratch'))
from model import ResNet50
from data import CatAndDogDataset

# --- Hyperparameters ---
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.2  # 20% of data for validation
CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_PATH = './best_model.pth'
RESULTS_FILE = './training_results.json'

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Instantiate model
model = ResNet50(num_classes=NUM_CLASSES, lr=LEARNING_RATE).to(device)
criterion = model.loss 
optimizer = model.configure_optimizers()
# Remove verbose parameter for compatibility
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Load dataset and split into train/validation
dataset = CatAndDogDataset(img_dir='../data/processed')
dataset_size = len(dataset)
val_size = int(VALIDATION_SPLIT * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Training samples: {train_size}, Validation samples: {val_size}")

# Track best validation accuracy
best_val_accuracy = 0.0
train_losses = []
val_losses = []
val_accuracies = []
train_accuracies = []
learning_rates = []

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / labels.size(0)

# Training loop
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch", colour="green")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        accuracy = calculate_accuracy(outputs, labels)
        running_loss += loss.item()
        running_accuracy += accuracy
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })
    
    # Calculate average training metrics for the epoch
    avg_train_loss = running_loss / len(train_dataloader)
    avg_train_accuracy = running_accuracy / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_accuracy = 0.0
    
    with torch.no_grad():
        val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", unit="batch", colour="blue")
        
        for images, labels in val_progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            accuracy = calculate_accuracy(outputs, labels)
            
            val_running_loss += loss.item()
            val_running_accuracy += accuracy
            
            val_progress_bar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_acc': f'{accuracy:.4f}'
            })
    
    # Calculate average validation metrics for the epoch
    avg_val_loss = val_running_loss / len(val_dataloader)
    avg_val_accuracy = val_running_accuracy / len(val_dataloader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)
    
    # Record current learning rate
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'train_accuracy': avg_train_accuracy,
        'val_loss': avg_val_loss,
        'val_accuracy': avg_val_accuracy,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_accuracy': avg_val_accuracy
        }, BEST_MODEL_PATH)
        print(f"✨ New best model saved with validation accuracy: {best_val_accuracy:.4f}")
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

print("🌟 Training finished! 🌟")

# Save training results
training_results = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'learning_rates': learning_rates,
    'best_val_accuracy': best_val_accuracy,
    'final_epoch': NUM_EPOCHS
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(training_results, f, indent=4)

# Plot training history
plt.figure(figsize=(15, 10))

# Plot loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot learning rate
plt.subplot(2, 2, 3)
plt.plot(learning_rates, label='Learning Rate', color='green')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)

# Plot best validation accuracy
plt.subplot(2, 2, 4)
plt.bar(['Best Validation Accuracy'], [best_val_accuracy], color='orange')
plt.title(f'Best Validation Accuracy: {best_val_accuracy:.4f}')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)

plt.tight_layout()
plt.savefig('./training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Training history plot saved as './training_history.png'")
print(f"Best model saved as '{BEST_MODEL_PATH}'")
print(f"Checkpoints saved in '{CHECKPOINT_DIR}/'")
print(f"Training results saved as '{RESULTS_FILE}'")

# Function to load and test the best model
def test_best_model():
    print("\nTesting the best model...")
    # Load the best model
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader, desc="Testing best model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Best model accuracy on validation set: {accuracy:.2f}%")
    
    return accuracy

# Test the best model
test_best_model()