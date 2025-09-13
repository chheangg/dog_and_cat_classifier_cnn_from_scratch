import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom modules
from model import ResNet50, Conv2D, LinearRegression, BatchNorm2d
from data import CatAndDogDataset
from torchvision import transforms

def debug_model_behavior():
    """Debug script to identify why model outputs similar probabilities"""
    
    print("🔍 Starting Model Debug Analysis...")
    print("=" * 60)
    
    # Setup
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Create a simple test dataset
    test_transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    # Instantiate model
    model = ResNet50(num_classes=2, lr=0.01, in_channels=3, dropout_rate=0.3).to(device)
    
    print("📊 Model Architecture Overview:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # 1. Check Initial Weight Statistics
    print("1. 📈 Initial Weight Statistics:")
    print("-" * 40)
    
    weight_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name or 'w' in name:
                weight_stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'abs_mean': param.data.abs().mean().item()
                }
                print(f"   {name:30} | mean: {weight_stats[name]['mean']:8.4f} | std: {weight_stats[name]['std']:8.4f} | range: [{weight_stats[name]['min']:8.4f}, {weight_stats[name]['max']:8.4f}]")
    
    print()
    
    # 2. Forward Pass Analysis
    print("2. 🔍 Forward Pass Analysis:")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        print(f"   Input images shape: {images.shape}")
        print(f"   Input range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"   Input mean: {images.mean().item():.3f}, std: {images.std().item():.3f}")
        print()
        
        # Monitor activations through layers
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks to monitor key layers
        hooks = []
        layer_names = ['conv1', 'conv2.0.conv1', 'conv2.0.conv3', 'conv3.0.conv1', 
                      'conv4.0.conv1', 'conv5.0.conv1', 'fc', 'softmax']
        
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Forward pass
        outputs = model(images)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print("   Layer-wise Activation Statistics:")
        for name, activation in activations.items():
            print(f"   {name:30} | mean: {activation.mean().item():8.4f} | std: {activation.std().item():8.4f} | range: [{activation.min().item():8.4f}, {activation.max().item():8.4f}]")
        
        print()
        print("   Final Output Analysis:")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output values: {outputs.cpu().numpy()}")
        print(f"   Output mean: {outputs.mean().item():.6f}")
        print(f"   Output std: {outputs.std().item():.6f}")
        print(f"   Softmax probabilities: {torch.softmax(outputs, dim=1).cpu().numpy()}")
        print(f"   Predictions: {torch.argmax(outputs, dim=1).cpu().numpy()}")
        print(f"   Labels: {labels.cpu().numpy()}")
    
    print()
    
    # 3. Gradient Flow Analysis (Backward Pass)
    print("3. 📉 Gradient Flow Analysis:")
    print("-" * 40)
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Forward + backward
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Analyze gradients
    print("   Gradient Statistics (grad_norm / param_norm ratio):")
    print("   (Ideal ratio: 0.001-0.1, >1 = exploding, <0.001 = vanishing)")
    print()
    
    gradient_issues = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            param_norm = param.data.norm(2).item() if param.data.numel() > 0 else 1e-8
            ratio = grad_norm / (param_norm + 1e-8)
            
            status = "NORMAL"
            if ratio > 1.0:
                status = "EXPLODING! 🚨"
                gradient_issues.append((name, ratio, "exploding"))
            elif ratio < 0.001:
                status = "VANISHING! ⚠️"
                gradient_issues.append((name, ratio, "vanishing"))
            
            print(f"   {name:40} | ratio: {ratio:10.4f} | {status}")
    
    print()
    
    # 4. Weight Update Analysis
    print("4. ⚖️ Weight Update Analysis:")
    print("-" * 40)
    
    # Store initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    # Perform one optimization step
    optimizer.step()
    optimizer.zero_grad()
    
    # Check weight changes
    print("   Weight changes after one update:")
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_weights:
            change = (param.data - initial_weights[name]).abs().mean().item()
            relative_change = change / (initial_weights[name].abs().mean().item() + 1e-8)
            
            status = "OK"
            if relative_change > 0.1:
                status = "LARGE UPDATE! ⚠️"
            elif relative_change < 1e-6:
                status = "TINY UPDATE! ⚠️"
            
            print(f"   {name:40} | change: {change:8.6f} | relative: {relative_change:8.6f} | {status}")
    
    print()
    
    # 5. Diagnosis Summary
    print("5. 🩺 Diagnosis Summary:")
    print("-" * 40)
    
    if gradient_issues:
        print("   🔴 GRADIENT ISSUES DETECTED:")
        exploding_count = sum(1 for _, _, issue_type in gradient_issues if issue_type == "exploding")
        vanishing_count = sum(1 for _, _, issue_type in gradient_issues if issue_type == "vanishing")
        
        print(f"   - Exploding gradients: {exploding_count} layers")
        print(f"   - Vanishing gradients: {vanishing_count} layers")
        
        # Show top 5 problematic layers
        gradient_issues.sort(key=lambda x: abs(x[1] - 0.01), reverse=True)
        print("   Top problematic layers:")
        for name, ratio, issue_type in gradient_issues[:5]:
            print(f"     {name}: ratio={ratio:.4f} ({issue_type})")
    else:
        print("   ✅ Gradient flow appears normal")
    
    # Check output diversity
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        prob_std = probs.std().item()
        
        print(f"\n   Output Probability Analysis:")
        print(f"   Probability std: {prob_std:.6f}")
        if prob_std < 0.01:
            print("   🔴 LOW OUTPUT DIVERSITY: Probabilities are too similar!")
            print("   Possible causes:")
            print("   - Exploding/vanishing gradients")
            print("   - Poor weight initialization")
            print("   - Learning rate too high")
            print("   - BatchNorm issues")
        else:
            print("   ✅ Output diversity appears normal")
    
    print()
    print("6. 💡 Recommended Fixes:")
    print("-" * 40)
    
    if gradient_issues:
        if exploding_count > vanishing_count:
            print("   🎯 PRIMARY ISSUE: Exploding Gradients")
            print("   Recommended fixes:")
            print("   - Add gradient clipping (torch.nn.utils.clip_grad_norm_)")
            print("   - Reduce learning rate (try 0.001 instead of 0.01)")
            print("   - Add proper weight initialization (Kaiming/He init)")
            print("   - Add L2 regularization")
        else:
            print("   🎯 PRIMARY ISSUE: Vanishing Gradients")
            print("   Recommended fixes:")
            print("   - Use ReLU instead of sigmoid/tanh")
            print("   - Add BatchNorm layers")
            print("   - Use residual connections")
            print("   - Increase learning rate")
    else:
        print("   ✅ No major gradient issues detected")
        print("   Try these general improvements:")
        print("   - Add proper weight initialization")
        print("   - Add learning rate scheduling")
        print("   - Check data preprocessing")
    
    return gradient_issues

def visualize_activations():
    """Visualize activations to understand information flow"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(num_classes=2, lr=0.01, in_channels=3, dropout_rate=0.3).to(device)
    model.eval()
    
    # Get sample data
    test_dataset = CatAndDogDataset(img_dir='../data/processed', train=True, 
                                  transform=transforms.Compose([
                                      transforms.Lambda(lambda x: x / 255.0),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225]),
                                  ]))
    
    images, _ = next(iter(DataLoader(test_dataset, batch_size=1)))
    images = images.to(device)
    
    # Hook to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    layers_to_monitor = ['conv1', 'conv2.0.conv3', 'conv3.0.conv3', 'conv4.0.conv3', 'conv5.0.conv3', 'fc']
    
    for name, module in model.named_modules():
        if any(layer in name for layer in layers_to_monitor):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot activations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (layer_name, activation) in enumerate(activations.items()):
        if i >= len(axes):
            break
            
        # Flatten and histogram
        flat_act = activation.cpu().numpy().flatten()
        axes[i].hist(flat_act, bins=50, alpha=0.7)
        axes[i].set_title(f'{layer_name}\nmean: {np.mean(flat_act):.3f}, std: {np.std(flat_act):.3f}')
        axes[i].set_xlabel('Activation Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('activation_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Activation histograms saved as 'activation_histograms.png'")

if __name__ == "__main__":
    # Run the debug analysis
    issues = debug_model_behavior()
    
    # Generate visualization
    print("\n📈 Generating activation visualizations...")
    visualize_activations()
    
    print("\n🎯 Debug completed! Check the output above for issues and recommendations.")
    print("💡 Run this script before training to identify potential problems.")