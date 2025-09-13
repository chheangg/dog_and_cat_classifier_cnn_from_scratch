import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import your custom modules
os.path.abspath(os.path.join(os.getcwd(), '..', 'dog_and_cat_classifier_cnn_from_scratch'))
from dog_and_cat_classifier_cnn_from_scratch.model import ResNet50
from dog_and_cat_classifier_cnn_from_scratch.data import CatAndDogDataset

# --- Hyperparameters ---
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.2  # 20% of data for validation
CHECKPOINT_DIR = '../models/checkpoints'
BEST_MODEL_PATH = '../models/best_model.pth'

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Instantiate model
model = ResNet50(num_classes=NUM_CLASSES, lr=LEARNING_RATE).to(device)
criterion = model.loss 
optimizer = model.configure_optimizers()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

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
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_accuracy': avg_val_accuracy
    }
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✨ New best model saved with validation accuracy: {best_val_accuracy:.4f}")
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
    print(f"Best Val Accuracy: {best_val_accuracy:.4f}\n")

print("🌟 Training finished! 🌟")

# Plot training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./training_history.png')
plt.show()

print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Training history plot saved as './training_history.png'")
print(f"Best model saved as '{BEST_MODEL_PATH}'")
print(f"Checkpoints saved in '{CHECKPOINT_DIR}/'")
print(f"TensorBoard logs available with: tensorboard --logdir=./runs")