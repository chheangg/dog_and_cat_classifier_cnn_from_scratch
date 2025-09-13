import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import gc
import os

# Import your custom modules
from dog_and_cat_classifier_cnn_from_scratch.model import ResNet50, Conv2D, LinearRegression, CrossEntropyError, softmax
from dog_and_cat_classifier_cnn_from_scratch.data import CatAndDogDataset

# --- Debug Configuration ---
DEBUG_BATCH_SIZE = 4
DEBUG_NUM_BATCHES = 5
DEBUG_LEARNING_RATE = 0.0001  # Very small for debugging

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def check_nan_inf(tensor, name="tensor"):
    """Check for NaN and Inf values in a tensor"""
    if torch.isnan(tensor).any():
        print(f"🚨 NaN detected in {name}!")
        return True
    if torch.isinf(tensor).any():
        print(f"🚨 Inf detected in {name}!")
        return True
    return False

def debug_forward_pass(model, dataloader):
    """Debug the forward pass step by step"""
    print("🔍 Debugging Forward Pass...")
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= DEBUG_NUM_BATCHES:
                break
                
            print(f"\n--- Batch {batch_idx + 1} ---")
            images, labels = images.cuda(), labels.cuda()
            
            # Forward pass with layer-by-layer debugging
            x = images
            print(f"Input shape: {x.shape}")
            
            # Manually go through each layer to check for issues
            layers_to_check = [
                ('conv1', model.conv1),
                ('bn1', model.bn1),
                ('relu', model.relu),
                ('pool1', model.pool1),
                ('dropout1', model.dropout1)
            ]
            
            for layer_name, layer in layers_to_check:
                try:
                    x = layer(x)
                    print(f"{layer_name}: {x.shape}, mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
                    
                    if check_nan_inf(x, layer_name):
                        print(f"❌ Problem at {layer_name}!")
                        return False
                        
                except Exception as e:
                    print(f"❌ Error in {layer_name}: {e}")
                    return False
            
            # Check final output
            try:
                output = model(images)
                print(f"Final output shape: {output.shape}")
                print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
                
                if check_nan_inf(output, "final_output"):
                    return False
                    
                # Check softmax probabilities
                probs = softmax(output)
                print(f"Softmax probs sum: {probs.sum(dim=1).mean().item():.6f}")
                
            except Exception as e:
                print(f"❌ Error in final forward pass: {e}")
                return False
                
    print("✅ Forward pass looks good!")
    return True

def debug_backward_pass(model, dataloader, criterion):
    """Debug the backward pass and gradients"""
    print("\n🔍 Debugging Backward Pass...")
    model.train()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= DEBUG_NUM_BATCHES:
            break
            
        print(f"\n--- Backward Batch {batch_idx + 1} ---")
        images, labels = images.cuda(), labels.cuda()
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        try:
            output = model(images)
            loss = criterion(output, labels)
            print(f"Loss: {loss.item():.6f}")
            
            if check_nan_inf(loss, "loss"):
                return False
                
        except Exception as e:
            print(f"❌ Forward pass error: {e}")
            return False
        
        # Backward pass
        try:
            loss.backward()
            print("✅ Backward pass completed")
        except Exception as e:
            print(f"❌ Backward pass error: {e}")
            return False
        
        # Check gradients
        total_params = 0
        nan_grad_params = 0
        inf_grad_params = 0
        large_grad_params = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_params += 1
                grad = param.grad
                
                if torch.isnan(grad).any():
                    nan_grad_params += 1
                    print(f"🚨 NaN gradient in {name}")
                
                if torch.isinf(grad).any():
                    inf_grad_params += 1
                    print(f"🚨 Inf gradient in {name}")
                
                if grad.abs().max() > 1000:  # Very large gradients
                    large_grad_params += 1
                    print(f"⚠️ Large gradient in {name}: {grad.abs().max().item():.6f}")
        
        print(f"Gradient stats - Total: {total_params}, NaN: {nan_grad_params}, Inf: {inf_grad_params}, Large: {large_grad_params}")
        
        if nan_grad_params > 0 or inf_grad_params > 0:
            return False
    
    print("✅ Backward pass looks good!")
    return True

def debug_optimizer_step(model, dataloader, criterion, optimizer):
    """Debug optimizer step"""
    print("\n🔍 Debugging Optimizer Step...")
    model.train()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 2:  # Just check a couple of steps
            break
            
        print(f"\n--- Optimizer Step {batch_idx + 1} ---")
        
        # Save initial weights
        initial_weights = {}
        for name, param in model.named_parameters():
            initial_weights[name] = param.data.clone()
        
        # Forward and backward
        images, labels = images.cuda(), labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        model.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Check weight updates
        print("Weight updates:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                update = param.data - initial_weights[name]
                update_norm = update.norm().item()
                print(f"  {name}: update norm = {update_norm:.6f}")
                
                if update_norm > 10:  # Very large update
                    print(f"  ⚠️ Large update in {name}")
    
    print("✅ Optimizer step looks good!")
    return True

def debug_data_pipeline():
    """Debug the data loading pipeline"""
    print("🔍 Debugging Data Pipeline...")
    
    # Simple transform for debugging
    debug_transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=debug_transform)
        print(f"Dataset size: {len(dataset)}")
        
        # Check a few samples
        for i in range(min(3, len(dataset))):
            image, label = dataset[i]
            print(f"Sample {i}: shape={image.shape}, label={label}, "
                  f"range=[{image.min().item():.3f}, {image.max().item():.3f}]")
            
            if check_nan_inf(image, f"sample_{i}"):
                return False
                
    except Exception as e:
        print(f"❌ Data pipeline error: {e}")
        return False
    
    print("✅ Data pipeline looks good!")
    return True

def debug_model_initialization():
    """Debug model initialization"""
    print("🔍 Debugging Model Initialization...")
    
    try:
        # Create a small test model
        test_model = ResNet50(num_classes=2, lr=DEBUG_LEARNING_RATE, in_channels=3, dropout_rate=0.3)
        test_model = test_model.cuda()
        
        # Check parameter initialization
        print("Parameter initialization stats:")
        for name, param in test_model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: mean={param.data.mean().item():.6f}, "
                      f"std={param.data.std().item():.6f}, "
                      f"shape={param.shape}")
                
                if check_nan_inf(param.data, name):
                    return False
        
        # Test a forward pass with random data
        test_input = torch.randn(DEBUG_BATCH_SIZE, 3, 224, 224).cuda()
        output = test_model(test_input)
        print(f"Test forward pass: output shape={output.shape}")
        
        if check_nan_inf(output, "test_output"):
            return False
            
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False
    
    print("✅ Model initialization looks good!")
    return True

def main():
    """Main debug function"""
    print("🐛 Starting Debug Session 🐛")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear memory
    clear_memory()
    
    # Create simple dataloader for debugging
    debug_transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    debug_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=debug_transform)
    debug_loader = DataLoader(debug_dataset, batch_size=DEBUG_BATCH_SIZE, shuffle=True)
    
    # Initialize model with debug settings
    model = ResNet50(num_classes=2, lr=DEBUG_LEARNING_RATE, in_channels=3, dropout_rate=0.3)
    model = model.to(device)
    
    # Use simpler optimizer for debugging
    optimizer = torch.optim.Adam(model.parameters(), lr=DEBUG_LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # Use built-in for debugging
    
    # Run debug steps
    debug_steps = [
        ("Data Pipeline", lambda: debug_data_pipeline()),
        ("Model Initialization", lambda: debug_model_initialization()),
        ("Forward Pass", lambda: debug_forward_pass(model, debug_loader)),
        ("Backward Pass", lambda: debug_backward_pass(model, debug_loader, criterion)),
        ("Optimizer Step", lambda: debug_optimizer_step(model, debug_loader, criterion, optimizer)),
    ]
    
    results = {}
    for step_name, debug_func in debug_steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            success = debug_func()
            results[step_name] = success
            if not success:
                print(f"❌ {step_name} FAILED!")
                break
        except Exception as e:
            print(f"❌ {step_name} CRASHED: {e}")
            results[step_name] = False
            break
    
    # Print summary
    print("\n" + "="*50)
    print("🐛 DEBUG SUMMARY 🐛")
    print("="*50)
    
    all_passed = True
    for step_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{step_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All debug tests passed! You can try training now.")
        print("Recommended next steps:")
        print("1. Use learning rate: 0.0001")
        print("2. Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
        print("3. Start with batch size: 16")
    else:
        print("\n🔧 Some debug tests failed. Check the specific error messages above.")
    
    return all_passed

if __name__ == "__main__":
    main()