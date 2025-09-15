import torch
import torch.nn as nn
from model import ResNet50, CrossEntropyError
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def kaiming_init(module):
    """Apply Kaiming initialization to Conv2D and Linear layers"""
    if hasattr(module, 'w') and hasattr(module.w, 'data'):
        nn.init.kaiming_normal_(module.w, mode='fan_out', nonlinearity='relu')
        if hasattr(module, 'b') and module.b is not None:
            nn.init.zeros_(module.b)

def test_learning_rates():
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Learning rates to test
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    # Create dummy dataset
    batch_size = 32
    num_batches = 10  # Simulate one epoch with 10 batches
    
    # Create dummy data loader
    dummy_inputs = [torch.randn(batch_size, 3, 224, 224).to(device) for _ in range(num_batches)]
    dummy_labels = [torch.randint(0, 2, (batch_size,)).to(device) for _ in range(num_batches)]
    
    results = {}
    
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Testing learning rate: {lr}")
        print(f"{'='*60}")
        
        # Create fresh model for each learning rate
        model = ResNet50(num_classes=2, lr=lr, in_channels=3, dropout_rate=0.3).to(device)
        
        # Apply Kaiming initialization
        model.apply(kaiming_init)
        
        # Get optimizer and criterion
        optimizer = model.configure_optimizers()
        criterion = CrossEntropyError
        
        # Track statistics during training
        batch_stats = {
            'loss': [],
            'output_min': [],
            'output_max': [],
            'output_mean': [],
            'output_std': [],
            'grad_norm': []
        }
        
        # Set model to training mode
        model.train()
        
        # Train for one epoch (multiple batches)
        for batch_idx in range(num_batches):
            # Get batch data
            inputs = dummy_inputs[batch_idx]
            labels = dummy_labels[batch_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Track gradient norms before update
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            probabilities = outputs
            batch_stats['loss'].append(loss.item())
            batch_stats['output_min'].append(probabilities.min().item())
            batch_stats['output_max'].append(probabilities.max().item())
            batch_stats['output_mean'].append(probabilities.mean().item())
            batch_stats['output_std'].append(probabilities.std().item())
            batch_stats['grad_norm'].append(total_grad_norm)
            
            # Print batch progress
            if (batch_idx + 1) % 2 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}:")
                print(f"    Loss: {loss.item():.6f}")
                print(f"    Outputs - Min: {probabilities.min().item():.6f}, "
                      f"Max: {probabilities.max().item():.6f}, "
                      f"Mean: {probabilities.mean().item():.6f}, "
                      f"Std: {probabilities.std().item():.6f}")
                print(f"    Gradient Norm: {total_grad_norm:.6f}")
        
        # Store results for this learning rate
        results[lr] = batch_stats
        
        # Print summary for this learning rate
        print(f"\n  Summary for LR={lr}:")
        print(f"    Final Loss: {batch_stats['loss'][-1]:.6f}")
        print(f"    Final Output Range: [{batch_stats['output_min'][-1]:.6f}, {batch_stats['output_max'][-1]:.6f}]")
        print(f"    Average Gradient Norm: {np.mean(batch_stats['grad_norm']):.6f}")
        
        # Check for exploding outputs
        final_max = batch_stats['output_max'][-1]
        if final_max > 0.99:
            print(f"    ⚠️  WARNING: Outputs approaching 1.0 (potential saturation)")
        elif final_max > 10.0:
            print(f"    🚨 EXPLODING OUTPUTS: Max output = {final_max:.6f}")
        
        # Check for vanishing outputs
        final_min = batch_stats['output_min'][-1]
        if final_min < 0.01 and final_max < 0.1:
            print(f"    ⚠️  WARNING: Outputs very small (potential vanishing)")
    
    # Plot results
    plot_results(results, learning_rates)

def plot_results(results, learning_rates):
    """Plot the training statistics for different learning rates"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Behavior with Different Learning Rates (1 Epoch)')
    
    metrics = ['loss', 'output_min', 'output_max', 'output_mean', 'output_std', 'grad_norm']
    titles = ['Loss', 'Output Min', 'Output Max', 'Output Mean', 'Output Std', 'Gradient Norm']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//3, i%3]
        for lr in learning_rates:
            ax.plot(results[lr][metric], label=f'LR={lr}')
        ax.set_title(title)
        ax.set_xlabel('Batch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_rate_debug.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    for lr in learning_rates:
        stats = results[lr]
        final_loss = stats['loss'][-1]
        avg_grad_norm = np.mean(stats['grad_norm'])
        output_range = stats['output_max'][-1] - stats['output_min'][-1]
        
        print(f"LR={lr}:")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Avg Gradient Norm: {avg_grad_norm:.6f}")
        print(f"  Output Range: {output_range:.6f}")
        
        if avg_grad_norm > 1000:
            print(f"  ❌ Too high - gradients exploding")
        elif avg_grad_norm < 0.001:
            print(f"  ❌ Too low - gradients vanishing")
        elif final_loss > 10.0:
            print(f"  ❌ Loss too high - learning rate may be too high")
        else:
            print(f"  ✅ Looks good - consider this learning rate")

if __name__ == "__main__":
    test_learning_rates()