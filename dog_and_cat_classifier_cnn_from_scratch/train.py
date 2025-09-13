import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
import gc
import os
import json
from datetime import datetime

# Import your custom dataset
from dog_and_cat_classifier_cnn_from_scratch.data import CatAndDogDataset

# --- Hyperparameters ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.2

def create_resnet_model(num_classes=2, pretrained=True, dropout_rate=0.3):
    """
    Create a ResNet50 model with custom classifier head
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for the classifier
    """
    # Load pretrained ResNet50
    model = models.resnet50(pretrained=pretrained)
    
    # Get the number of features from the last layer
    num_features = model.fc.in_features
    
    # Replace the classifier with our custom head
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )
    
    return model

def log_weight_stats(model):
    """Print mean and std for each layer"""
    print("📊 Weight Statistics:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                w_mean = module.weight.mean().item()
                w_std = module.weight.std().item()
                print(f" - {name}: mean={w_mean:.4f}, std={w_std:.4f}")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving
os.makedirs('../models', exist_ok=True)
os.makedirs('../models/training_checkpoints', exist_ok=True)

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

clear_memory()

# --- Automatic Model Loading ---
def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    checkpoints = [f for f in os.listdir('../models/training_checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join('../models/training_checkpoints', x)), reverse=True)
    return os.path.join('../models/training_checkpoints', checkpoints[0])

def load_checkpoint(model, optimizer=None):
    """Load model from checkpoint"""
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        training_history = checkpoint['training_history']
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"🔄 Resuming from epoch {start_epoch}")
        return start_epoch, best_val_loss, training_history
    
    print("🚀 No checkpoint found, starting fresh training")
    return 0, float('inf'), []

# --- Automatic Model Saving ---
def save_checkpoint(epoch, model, optimizer, best_val_loss, training_history, is_best=False):
    """Save model checkpoint"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'training_history': training_history,
        'timestamp': timestamp,
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS
        }
    }

    # Save regular checkpoint
    checkpoint_path = f'../models/training_checkpoints/checkpoint_epoch_{epoch+1}_{timestamp}.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save as best model if it's the best so far
    if is_best:
        best_model_path = f'../models/best_model_{timestamp}.pth'
        torch.save(checkpoint, best_model_path)
        print(f"💾 Saved best model: {best_model_path}")

    # Also save training history as JSON for easy analysis
    history_path = f'../models/training_checkpoints/training_history.json'
    # Convert any Tensors in training_history to standard Python types before saving
    serializable_history = [{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in training_history]
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)

    return checkpoint_path

def save_final_model(model, training_history, final_val_loss, final_val_acc):
    """Save final model after training completes"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'final_val_loss': final_val_loss,
        'final_val_acc': final_val_acc,
        'training_history': training_history,
        'timestamp': timestamp,
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS
        }
    }
    
    final_path = f'../models/final_model_{timestamp}.pth'
    torch.save(final_checkpoint, final_path)
    print(f"🎯 Saved final model: {final_path}")
    return final_path

def adjust_learning_rate(optimizer, epoch, base_lr=LEARNING_RATE, schedule=[20, 35], gamma=0.1):
    """
    Reduce learning rate at specific epochs.
    
    schedule: list of epochs to decay at
    gamma: decay factor
    """
    lr = base_lr
    for milestone in schedule:
        if epoch >= milestone:
            lr *= gamma
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

# --- Create Model ---
print("🏗️ Creating ResNet50 model...")
model = create_resnet_model(num_classes=NUM_CLASSES, pretrained=False, dropout_rate=0.3)
model = model.to(device)

# Initialize only the new classifier layers (if using pretrained weights)
def init_classifier_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply initialization only to the classifier
model.fc.apply(init_classifier_weights)

# Log initial weight statistics
log_weight_stats(model)

# --- Optimizer and Loss Function ---
# Use AdamW for better performance with ResNet
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# Alternative: SGD with momentum
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

print(f"✅ Using AdamW optimizer and CrossEntropyLoss")

# Load checkpoint if exists
start_epoch, best_val_loss, training_history = load_checkpoint(model, optimizer)

# --- Dataset and DataLoaders ---
# Define separate transforms for training and validation
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create two separate datasets with the correct transforms
train_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=train_transform)
val_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=val_transform)

# Determine the split sizes
dataset_size = len(train_dataset)
val_size = int(VALIDATION_SPLIT * dataset_size)
train_size = dataset_size - val_size

# Create a list of indices for splitting
indices = torch.randperm(dataset_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create subsets from the new indices
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(val_dataset, val_indices)

print(f"📊 Training: {len(train_subset)} samples")
print(f"📊 Validation: {len(val_subset)} samples")

train_loader = DataLoader(train_subset, 
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=4)

val_loader = DataLoader(val_subset,
                       batch_size=BATCH_SIZE * 2,
                       shuffle=False,
                       pin_memory=True,
                       num_workers=2)

# --- Training Loop ---
print(f"\n🚀 Starting training from epoch {start_epoch + 1}...")
for epoch in range(start_epoch, NUM_EPOCHS):
    # --- Adjust learning rate ---
    current_lr = adjust_learning_rate(optimizer, epoch)
    print(f"🔧 Epoch {epoch+1} | Learning Rate: {current_lr:.5f}")
    
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Pure loss for logging
        pure_loss = loss.item()
        train_loss += pure_loss * images.size(0)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{pure_loss:.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'mem': f'{torch.cuda.memory_allocated()/1024**3:.2f}GB',
        })
    
    # --- Validation ---
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    avg_train_loss = train_loss / len(train_subset)
    train_acc = 100. * correct / total
    avg_val_loss = val_loss / len(val_subset)
    val_acc = 100. * val_correct / val_total
    
    # --- Save epoch stats ---
    epoch_stats = {
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'train_acc': train_acc,
        'val_loss': avg_val_loss,
        'val_acc': val_acc,
        'learning_rate': current_lr,
        'timestamp': datetime.now().isoformat(),
    }
    training_history.append(epoch_stats)
    
    print(f"\n📊 Epoch {epoch+1}:")
    print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Log weight stats (less frequently to avoid clutter)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        log_weight_stats(model)
    
    # Check best model
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        print("   🎯 New best model!")
    
    # Save checkpoint
    checkpoint_path = save_checkpoint(epoch, model, optimizer, best_val_loss, training_history, is_best)
    print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    clear_memory()

# --- Final Model Save ---
print(f"\n🎯 Training completed!")
print(f"   Best Validation Loss: {best_val_loss:.4f}")
print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"   Total Epochs Trained: {len(training_history)}")

final_path = save_final_model(model, training_history, best_val_loss, best_val_acc)
print("🌟 Training finished! 🌟")
print(f"📁 Models saved in: models/")
print(f"📁 Checkpoints saved in: models/training_checkpoints")