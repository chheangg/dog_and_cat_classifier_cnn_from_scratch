import os
import gc
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- Import your custom modules ---
from model import ResNet50, L2Regularization, BatchNorm2d, Conv2D, LinearRegression
from data import CatAndDogDataset

# --- Memory Optimization Setup ---
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True

# --- Hyperparameters ---
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 2
GRADIENT_ACCUMULATION_STEPS = 16
VALIDATION_SPLIT = 0.2

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# --- Create directories ---
os.makedirs('../models', exist_ok=True)
os.makedirs('../models/training_checkpoints', exist_ok=True)

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

clear_memory()

# --- Weight Initialization ---
def initialize_weights(model):
    """Apply Kaiming initialization to custom layers"""
    print("🔧 Applying Kaiming weight initialization...")
    for name, module in model.named_modules():
        if hasattr(module, 'w') and hasattr(module.w, 'data'):
            if isinstance(module, (Conv2D, LinearRegression)):
                nn.init.kaiming_normal_(module.w.data, mode='fan_out', nonlinearity='relu')
                if hasattr(module, 'b') and module.b is not None:
                    nn.init.constant_(module.b.data, 0)
        elif hasattr(module, 'gamma') and hasattr(module.gamma, 'data'):
            if isinstance(module, BatchNorm2d):
                nn.init.constant_(module.gamma.data, 1)
                nn.init.constant_(module.beta.data, 0)

# --- Instantiate Model ---
model = ResNet50(num_classes=NUM_CLASSES, lr=LEARNING_RATE, in_channels=3, dropout_rate=0.3).to(device)
initialize_weights(model)

# --- Optimizer and loss ---
optimizer = model.configure_optimizers()
criterion = model.loss
print(f"✅ Using model's built-in optimizer and loss function")

# --- Checkpoint Utilities ---
def find_latest_checkpoint():
    checkpoints = [f for f in os.listdir('../models/training_checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join('../models/training_checkpoints', x)), reverse=True)
    return os.path.join('../models/training_checkpoints', checkpoints[0])

def load_checkpoint(model, optimizer=None):
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        training_history = checkpoint['training_history']
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return start_epoch, best_val_loss, training_history
    return 0, float('inf'), []

def save_checkpoint(epoch, model, optimizer, best_val_loss, training_history, is_best=False):
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
    checkpoint_path = f'../models/training_checkpoints/checkpoint_epoch_{epoch+1}_{timestamp}.pth'
    torch.save(checkpoint, checkpoint_path)
    if is_best:
        best_model_path = f'../models/best_model_{timestamp}.pth'
        torch.save(checkpoint, best_model_path)
    history_path = f'../models/training_checkpoints/training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    return checkpoint_path

def save_final_model(model, training_history, final_val_loss, final_val_acc):
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
    return final_path

# --- Load checkpoint if exists ---
start_epoch, best_val_loss, training_history = load_checkpoint(model, optimizer)

# --- Mixed precision ---
scaler = torch.cuda.amp.GradScaler()

# --- Dataset and DataLoaders ---
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

train_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=train_transform)
val_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=val_transform)

dataset_size = len(train_dataset)
val_size = int(VALIDATION_SPLIT * dataset_size)
train_size = dataset_size - val_size

indices = torch.randperm(dataset_size).tolist()
train_subset = torch.utils.data.Subset(train_dataset, indices[:train_size])
val_subset = torch.utils.data.Subset(val_dataset, indices[train_size:])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True, num_workers=2)

print(f"📊 Training: {len(train_subset)} samples")
print(f"📊 Validation: {len(val_subset)} samples")
print(f"\n🚀 Starting training from epoch {start_epoch + 1}...")

# --- Training Loop ---
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            l2_lambda = 1e-4
            l2_reg = sum(L2Regularization(p, l2_lambda) for p in model.parameters())
            loss = (loss + l2_reg) / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if batch_idx % 20 == 0:
                clear_memory()

        pure_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS - (l2_lambda/2) * l2_reg.item()/GRADIENT_ACCUMULATION_STEPS
        train_loss += pure_loss * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': f'{pure_loss:.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'mem': f'{torch.cuda.memory_allocated()/1024**3:.2f}GB',
            'L2': f'{(l2_lambda/2)*l2_reg.item():.6f}',
        })

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    avg_train_loss = train_loss / len(train_dataset)
    train_acc = 100. * correct / total
    avg_val_loss = val_loss / len(val_dataset)
    val_acc = 100. * val_correct / val_total

    epoch_stats = {
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'train_acc': train_acc,
        'val_loss': avg_val_loss,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat(),
        'l2_lambda': l2_lambda,
        'learning_rate': LEARNING_RATE
    }
    training_history.append(epoch_stats)

    print(f"\n📊 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"             Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"             L2 λ={l2_lambda}, LR={LEARNING_RATE}")

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        print("   🎯 New best model!")

    checkpoint_path = save_checkpoint(epoch, model, optimizer, best_val_loss, training_history, is_best)
    print(f"   💾 Checkpoint saved: {checkpoint_path}")

    clear_memory()
    torch.cuda.reset_peak_memory_stats()

# --- Save final model ---
final_path = save_final_model(model, training_history, best_val_loss, best_val_acc)
print("🌟 Training finished! 🌟")
print(f"📁 Models")