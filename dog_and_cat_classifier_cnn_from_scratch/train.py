import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from model import ResNet50
from data import CatAndDogDataset

# --- Hyperparameters ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 8
NUM_CLASSES = 2 


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: mps 🚀")
else:
    device = torch.device("cpu")
    print("MPS device not found. Using device: cpu 🐌")
    
# Instantiate model, dataset, and dataloader
transform = transforms.ToTensor()
model = ResNet50(num_classes=NUM_CLASSES, lr=LEARNING_RATE, in_channels=3).to(device)

# Instantiate the loss function and optimizer
criterion = model.loss
optimizer = model.configure_optimizers() 

# Instantiate datasets and dataloaders
dataset = CatAndDogDataset(img_dir='../data/processed')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    # Set the model to training mode
    model.train()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch", colour="green")
    
    # Loop over each batch in the dataloader
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move data to the correct device
        images, labels = images.to(device), labels.to(device)    
        
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Forward pass: get predictions and calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Update the progress bar with the current loss
        # The .item() is needed to get the scalar value from the tensor
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
    print(f"\n✨ Epoch {epoch+1} completed! Average loss: {loss.item():.4f}\n")

print("🌟 Training finished! 🌟")