import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import gc
import os
import json
import numpy as np
from datetime import datetime
from torchvision import transforms
from torch import nn

os.path.abspath(os.path.join(os.getcwd(), '..', 'dog_and_cat_classifier_cnn_from_scratch'))

from dog_and_cat_classifier_cnn_from_scratch.model import ResNet50, Conv2D, LinearRegression
from dog_and_cat_classifier_cnn_from_scratch.data import CatAndDogDataset

# --- Hyperparameters ---
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.2

def kaiming_init(module):
    """Apply Kaiming initialization to Conv2D and Linear layers"""
    if isinstance(module, Conv2D):
        # He normal initialization for conv layers
        nn.init.kaiming_normal_(module.w, mode='fan_out', nonlinearity='relu')
        if module.b is not None:
            nn.init.zeros_(module.b)
    elif isinstance(module, LinearRegression):
        nn.init.kaiming_normal_(module.w, mode='fan_out', nonlinearity='linear')
        if module.b is not None:
            nn.init.zeros_(module.b)

def log_weight_stats(model):
    """Print mean and std for each Conv2D and Linear layer"""
    print("📊 Weight Statistics:")
    for name, module in model.named_modules():
        if isinstance(module, (Conv2D, LinearRegression)):
            w_mean = module.w.mean().item()
            w_std = module.w.std().item()
            print(f" - {name}: mean={w_mean:.4f}, std={w_std:.4f}")

def dump_weights_to_file(model, filepath):
    """Dump all model weights to a text file"""
    with open(filepath, 'w') as f:
        f.write("MODEL WEIGHTS DUMP\n")
        f.write("="*50 + "\n\n")
        
        for name, module in model.named_modules():
            if isinstance(module, (Conv2D, LinearRegression)):
                f.write(f"Layer: {name}\n")
                f.write(f"Weight shape: {module.w.shape}\n")
                f.write(f"Weight mean: {module.w.mean().item():.6f}\n")
                f.write(f"Weight std: {module.w.std().item():.6f}\n")
                f.write(f"Weight min: {module.w.min().item():.6f}\n")
                f.write(f"Weight max: {module.w.max().item():.6f}\n")
                
                # Dump actual weight values (first few and statistics)
                weights_flat = module.w.detach().cpu().numpy().flatten()
                f.write(f"First 20 weight values: {weights_flat[:20].tolist()}\n")
                
                if module.b is not None:
                    f.write(f"Bias shape: {module.b.shape}\n")
                    f.write(f"Bias values: {module.b.detach().cpu().numpy().tolist()}\n")
                
                f.write("-" * 30 + "\n\n")

def dump_image_and_output(image_batch, output_batch, labels_batch, filepath):
    """Dump first image and model output to text file"""
    with open(filepath, 'w') as f:
        f.write("IMAGE AND OUTPUT DUMP\n")
        f.write("="*50 + "\n\n")
        
        # First image info
        first_image = image_batch[0].detach().cpu().numpy()
        first_label = labels_batch[0].item()
        first_output = output_batch[0].detach().cpu().numpy()
        
        f.write(f"Image shape: {first_image.shape}\n")
        f.write(f"True label: {first_label}\n")
        f.write(f"Image min: {first_image.min():.6f}\n")
        f.write(f"Image max: {first_image.max():.6f}\n")
        f.write(f"Image mean: {first_image.mean():.6f}\n")
        f.write(f"Image std: {first_image.std():.6f}\n\n")
        
        # Sample pixel values from each channel
        f.write("Sample pixel values (first 10x10 from each channel):\n")
        for c in range(first_image.shape[0]):  # channels
            f.write(f"Channel {c}:\n")
            sample_pixels = first_image[c, :10, :10]
            for row in sample_pixels:
                f.write(" ".join(f"{pixel:8.4f}" for pixel in row) + "\n")
            f.write("\n")
        
        # Model output
        f.write(f"Raw model output: {first_output}\n")
        f.write(f"Output shape: {first_output.shape}\n")
        
        # Apply softmax to get probabilities
        softmax_output = torch.softmax(torch.tensor(first_output), dim=0).numpy()
        f.write(f"Softmax probabilities: {softmax_output}\n")
        f.write(f"Predicted class: {np.argmax(softmax_output)}\n")
        f.write(f"Prediction confidence: {np.max(softmax_output):.4f}\n")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving
os.makedirs('../models', exist_ok=True)
os.makedirs('../debug_dumps', exist_ok=True)

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

clear_memory()

# --- Load Specific Model ---
def load_specific_model(model_path):
    """Load specific model checkpoint"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    print(f"📂 Loading specific model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract information from checkpoint
    epoch = checkpoint.get('epoch', 'Unknown')
    best_val_loss = checkpoint.get('best_val_loss', 'Unknown')
    training_history = checkpoint.get('training_history', [])
    
    print(f"🔄 Model from epoch {epoch}, best val loss: {best_val_loss}")
    return checkpoint['model_state_dict'], epoch, best_val_loss

# --- Instantiate Model ---
model = ResNet50(num_classes=NUM_CLASSES, lr=LEARNING_RATE, in_channels=3, dropout_rate=0.3).to(device)

# Load the specific model
model_path = '../models/best_model_20250913_163301.pth'
state_dict, epoch_info, val_loss_info = load_specific_model(model_path)

if state_dict is None:
    print("❌ Failed to load model. Exiting.")
    exit(1)

model.load_state_dict(state_dict)
print(f"✅ Model loaded successfully!")

# Log weight statistics
log_weight_stats(model)

# Use model's own optimizer and loss function
optimizer = model.configure_optimizers()
criterion = model.loss

print(f"✅ Using model's built-in optimizer and loss function")

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

# --- DEBUG: Process first batch and dump info ---
print(f"\n🔍 DEBUG MODE: Processing first batch...")

model.eval()  # Set to eval mode for debugging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with torch.no_grad():
    # Get first batch from training data
    first_batch = next(iter(train_loader))
    images, labels = first_batch
    images, labels = images.to(device), labels.to(device)
    
    print(f"📊 Batch info:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels: {labels[:10].cpu().numpy()}")  # First 10 labels
    
    # Forward pass
    outputs = model(images)
    
    print(f"📊 Model outputs shape: {outputs.shape}")
    print(f"   First output: {outputs[0].cpu().numpy()}")
    
    # Dump weights
    weights_file = f'../debug_dumps/weights_dump_{timestamp}.txt'
    dump_weights_to_file(model, weights_file)
    print(f"💾 Weights dumped to: {weights_file}")
    
    # Dump first image and output
    image_output_file = f'../debug_dumps/image_output_dump_{timestamp}.txt'
    dump_image_and_output(images, outputs, labels, image_output_file)
    print(f"💾 Image and output dumped to: {image_output_file}")
    
    # Calculate loss for the batch
    loss = criterion(outputs, labels)
    print(f"📊 Batch loss: {loss.item():.6f}")
    
    # Calculate accuracy
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    accuracy = 100. * correct / labels.size(0)
    print(f"📊 Batch accuracy: {accuracy:.2f}%")
    
    # Show predictions vs actual for first 10 samples
    print(f"\n🎯 First 10 predictions vs actual:")
    for i in range(min(10, len(labels))):
        pred_prob = torch.softmax(outputs[i], dim=0)
        print(f"   Sample {i}: Actual={labels[i].item()}, Predicted={predicted[i].item()}, Confidence={pred_prob.max().item():.4f}")

print(f"\n🛑 DEBUG COMPLETE - EXITING")
print(f"📁 Debug files saved in: debug_dumps/")
print(f"   - {weights_file}")
print(f"   - {image_output_file}")