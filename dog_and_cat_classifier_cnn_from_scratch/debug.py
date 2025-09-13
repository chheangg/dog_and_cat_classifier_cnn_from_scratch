#!/usr/bin/env python3
"""
Comprehensive Debug Script for Vanishing Gradients in Deep Learning
Analyzes weight initialization, gradient flow, activation distributions, and loss behavior
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader

# Import your custom modules (assuming they're available)
from dog_and_cat_classifier_cnn_from_scratch.model import ResNet50
from dog_and_cat_classifier_cnn_from_scratch.data import CatAndDogDataset

class GradientDebugger:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gradient_norms = defaultdict(list)
        self.weight_norms = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.layer_names = []
        self.hooks = []
        
        # Register hooks for all layers
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks for gradient and activation monitoring"""
        def activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_stats[f"{name}_mean"].append(output.detach().mean().cpu().item())
                    self.activation_stats[f"{name}_std"].append(output.detach().std().cpu().item())
                    self.activation_stats[f"{name}_max"].append(output.detach().max().cpu().item())
                    self.activation_stats[f"{name}_min"].append(output.detach().min().cpu().item())
                    
                    # Check for dead neurons (all zeros)
                    zeros = (output.detach() == 0).float().mean().cpu().item()
                    self.activation_stats[f"{name}_dead_neurons"].append(zeros)
            return hook
        
        def gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norm = grad_output[0].detach().norm().cpu().item()
                    self.gradient_norms[name].append(grad_norm)
            return hook
        
        # Register hooks for all named modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                self.layer_names.append(name)
                # Forward hook for activations
                handle1 = module.register_forward_hook(activation_hook(name))
                # Backward hook for gradients
                handle2 = module.register_backward_hook(gradient_hook(name))
                self.hooks.extend([handle1, handle2])
    
    def check_weight_initialization(self):
        """Analyze weight initialization across all layers"""
        print("=" * 60)
        print("WEIGHT INITIALIZATION ANALYSIS")
        print("=" * 60)
        
        init_stats = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name or 'w' in name:
                weight_data = param.data.cpu().numpy()
                
                stats = {
                    'mean': float(np.mean(weight_data)),
                    'std': float(np.std(weight_data)),
                    'min': float(np.min(weight_data)),
                    'max': float(np.max(weight_data)),
                    'shape': list(weight_data.shape),
                    'total_params': int(np.prod(weight_data.shape))
                }
                
                init_stats[name] = stats
                
                print(f"\n{name}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Mean: {stats['mean']:.6f}")
                print(f"  Std:  {stats['std']:.6f}")
                print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                
                # Check if initialization looks reasonable
                if stats['std'] < 0.001:
                    print(f"  ⚠️  WARNING: Very small std deviation - possible vanishing gradient risk")
                elif stats['std'] > 1.0:
                    print(f"  ⚠️  WARNING: Very large std deviation - possible exploding gradient risk")
                else:
                    print(f"  ✅ Initialization looks reasonable")
        
        return init_stats
    
    def analyze_forward_pass(self, data_loader, num_batches=3):
        """Analyze forward pass behavior"""
        print("\n" + "=" * 60)
        print("FORWARD PASS ANALYSIS")
        print("=" * 60)
        
        self.model.eval()
        
        logits_all = []
        predictions_all = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                logits_all.append(outputs.cpu().numpy())
                
                # Get predictions
                _, predicted = outputs.max(1)
                predictions_all.append(predicted.cpu().numpy())
                
                print(f"\nBatch {batch_idx + 1}:")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Output range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
                print(f"  Output mean: {outputs.mean().item():.6f}")
                print(f"  Output std: {outputs.std().item():.6f}")
                
                # Check if outputs are too close to zero
                near_zero = (torch.abs(outputs) < 1e-6).float().mean().item()
                print(f"  % outputs near zero: {near_zero * 100:.2f}%")
                
                # Check prediction distribution
                unique, counts = torch.unique(predicted, return_counts=True)
                print(f"  Predictions: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")
        
        # Analyze overall statistics
        all_logits = np.concatenate(logits_all, axis=0)
        all_predictions = np.concatenate(predictions_all, axis=0)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total samples analyzed: {len(all_logits)}")
        print(f"  Logit statistics:")
        print(f"    Mean: {np.mean(all_logits):.6f}")
        print(f"    Std: {np.std(all_logits):.6f}")
        print(f"    Min: {np.min(all_logits):.6f}")
        print(f"    Max: {np.max(all_logits):.6f}")
        
        # Prediction distribution
        unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
        print(f"  Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
        
        return all_logits, all_predictions
    
    def analyze_backward_pass(self, data_loader, criterion, num_batches=3):
        """Analyze backward pass and gradient flow"""
        print("\n" + "=" * 60)
        print("BACKWARD PASS ANALYSIS")
        print("=" * 60)
        
        self.model.train()
        
        losses = []
        
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Clear previous gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Analyze gradients
            gradient_norms = {}
            total_gradient_norm = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    gradient_norms[name] = grad_norm
                    total_gradient_norm += grad_norm ** 2
                    
                    if grad_norm < 1e-7:
                        print(f"  ⚠️  {name}: Very small gradient norm {grad_norm:.2e}")
                    elif grad_norm > 1e2:
                        print(f"  ⚠️  {name}: Very large gradient norm {grad_norm:.2e}")
            
            total_gradient_norm = total_gradient_norm ** 0.5
            print(f"  Total gradient norm: {total_gradient_norm:.6f}")
            
            # Check for vanishing gradients in early layers
            early_layer_grads = [grad_norm for name, grad_norm in gradient_norms.items() 
                               if any(early in name for early in ['conv1', 'bn1', 'conv2.0'])]
            
            if early_layer_grads:
                min_early_grad = min(early_layer_grads)
                max_early_grad = max(early_layer_grads)
                print(f"  Early layer gradient range: [{min_early_grad:.2e}, {max_early_grad:.2e}]")
                
                if min_early_grad < 1e-6:
                    print(f"  ⚠️  WARNING: Vanishing gradients detected in early layers!")
        
        print(f"\nLoss statistics over {num_batches} batches:")
        print(f"  Mean loss: {np.mean(losses):.6f}")
        print(f"  Loss std: {np.std(losses):.6f}")
        print(f"  Loss range: [{np.min(losses):.6f}, {np.max(losses):.6f}]")
        
        return losses, gradient_norms
    
    def check_activation_flow(self):
        """Analyze activation statistics collected during forward pass"""
        if not self.activation_stats:
            print("No activation statistics collected. Run analyze_forward_pass first.")
            return
        
        print("\n" + "=" * 60)
        print("ACTIVATION FLOW ANALYSIS")
        print("=" * 60)
        
        for layer_name in self.layer_names:
            mean_key = f"{layer_name}_mean"
            std_key = f"{layer_name}_std"
            dead_key = f"{layer_name}_dead_neurons"
            
            if mean_key in self.activation_stats:
                mean_vals = self.activation_stats[mean_key]
                std_vals = self.activation_stats[std_key]
                dead_vals = self.activation_stats[dead_key]
                
                avg_mean = np.mean(mean_vals)
                avg_std = np.mean(std_vals)
                avg_dead = np.mean(dead_vals)
                
                print(f"\n{layer_name}:")
                print(f"  Average activation mean: {avg_mean:.6f}")
                print(f"  Average activation std: {avg_std:.6f}")
                print(f"  Average % dead neurons: {avg_dead * 100:.2f}%")
                
                if avg_std < 1e-6:
                    print(f"  ⚠️  WARNING: Very small activation variance - possible saturation")
                if avg_dead > 0.5:
                    print(f"  ⚠️  WARNING: High percentage of dead neurons")
                if abs(avg_mean) > 10:
                    print(f"  ⚠️  WARNING: Very large activation magnitudes")
    
    def diagnose_common_issues(self):
        """Diagnose common causes of vanishing gradients"""
        print("\n" + "=" * 60)
        print("COMMON ISSUES DIAGNOSIS")
        print("=" * 60)
        
        issues_found = []
        
        # Check model depth
        total_layers = len([name for name, _ in self.model.named_modules() 
                           if len(list(_.children())) == 0])
        print(f"Total layers in model: {total_layers}")
        if total_layers > 50:
            issues_found.append("Very deep network - high risk of vanishing gradients")
        
        # Check for custom vs PyTorch implementations
        has_custom_layers = any('Conv2D' in str(type(module)) or 'LinearRegression' in str(type(module))
                               for name, module in self.model.named_modules())
        if has_custom_layers:
            issues_found.append("Using custom layer implementations - verify gradient computation")
        
        # Check activation functions
        has_relu_only = True
        for name, module in self.model.named_modules():
            if 'activation' in name.lower() or 'sigmoid' in str(type(module)).lower():
                has_relu_only = False
                break
        
        if not has_relu_only:
            issues_found.append("Using saturating activations - may cause vanishing gradients")
        
        print(f"\nPotential issues found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        
        if not issues_found:
            print("  No obvious structural issues detected.")
        
        return issues_found
    
    def recommend_fixes(self, gradient_norms, init_stats):
        """Provide specific recommendations based on analysis"""
        print("\n" + "=" * 60)
        print("RECOMMENDED FIXES")
        print("=" * 60)
        
        recommendations = []
        
        # Check gradient magnitudes
        if gradient_norms:
            min_grad = min(gradient_norms.values())
            max_grad = max(gradient_norms.values())
            
            if min_grad < 1e-7:
                recommendations.append(
                    "1. VANISHING GRADIENTS: Consider using gradient clipping, "
                    "residual connections, or better initialization"
                )
            
            if max_grad > 1e2:
                recommendations.append(
                    "2. EXPLODING GRADIENTS: Use gradient clipping with max_norm=1.0"
                )
        
        # Check initialization
        small_init_layers = [name for name, stats in init_stats.items() 
                           if stats['std'] < 0.001]
        if small_init_layers:
            recommendations.append(
                f"3. POOR INITIALIZATION: Re-initialize these layers with proper scaling: "
                f"{small_init_layers[:3]}{'...' if len(small_init_layers) > 3 else ''}"
            )
        
        # Learning rate recommendations
        recommendations.append(
            "4. LEARNING RATE: Try starting with lr=0.001 instead of 0.1 for deep networks"
        )
        
        recommendations.append(
            "5. OPTIMIZER: Consider using Adam or SGD with momentum for better convergence"
        )
        
        recommendations.append(
            "6. BATCH NORMALIZATION: Ensure BatchNorm is working correctly and not in eval mode during training"
        )
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")
        
        return recommendations
    
    def save_debug_report(self, output_dir='debug_reports'):
        """Save comprehensive debug report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'gradient_norms': dict(self.gradient_norms),
            'weight_norms': dict(self.weight_norms),
            'activation_stats': dict(self.activation_stats),
            'layer_names': self.layer_names
        }
        
        report_path = os.path.join(output_dir, f'debug_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Debug report saved to: {report_path}")
        return report_path
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def run_comprehensive_debug(model, train_loader, device='cuda'):
    """Run complete debugging pipeline"""
    print("🔍 Starting Comprehensive Deep Learning Debug Analysis")
    print("=" * 80)
    
    # Initialize debugger
    debugger = GradientDebugger(model, device)
    
    try:
        # 1. Check weight initialization
        init_stats = debugger.check_weight_initialization()
        
        # 2. Analyze forward pass
        logits, predictions = debugger.analyze_forward_pass(train_loader)
        
        # 3. Analyze backward pass
        criterion = model.loss  # Use your custom loss
        losses, gradient_norms = debugger.analyze_backward_pass(train_loader, criterion)
        
        # 4. Check activation flow
        debugger.check_activation_flow()
        
        # 5. Diagnose common issues
        issues = debugger.diagnose_common_issues()
        
        # 6. Provide recommendations
        recommendations = debugger.recommend_fixes(gradient_norms, init_stats)
        
        # 7. Save report
        debugger.save_debug_report()
        
        print("\n" + "=" * 80)
        print("🎯 DEBUG ANALYSIS COMPLETE")
        print("=" * 80)
        
        return {
            'init_stats': init_stats,
            'logits': logits,
            'predictions': predictions,
            'losses': losses,
            'gradient_norms': gradient_norms,
            'issues': issues,
            'recommendations': recommendations
        }
        
    finally:
        # Clean up hooks
        debugger.cleanup()

def kaiming_init_linear(module):
    """Initialize Linear layers with Kaiming initialization"""
    if hasattr(module, 'w') and module.w is not None:
        nn.init.kaiming_normal_(module.w, mode='fan_out', nonlinearity='relu')
    if hasattr(module, 'b') and module.b is not None:
        nn.init.constant_(module.b, 0)

def kaiming_init_conv2d(module):
    """Initialize Conv2D layers with Kaiming initialization"""
    if hasattr(module, 'w') and module.w is not None:
        nn.init.kaiming_normal_(module.w, mode='fan_out', nonlinearity='relu')
    if hasattr(module, 'b') and module.b is not None:
        nn.init.constant_(module.b, 0)
        
# Example usage function
def debug_your_model():
    """
    Example of how to use the debugger with your specific model
    Uncomment and modify paths as needed
    """
    
    # Setup (uncomment and modify as needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load your model
    model = ResNet50(num_classes=2, lr=0.01, in_channels=3, dropout_rate=0.3).to(device)
    
    # Apply Kaiming initialization (your existing code)
    for module in model.modules():
        if hasattr(module, '__class__'):
            class_name = module.__class__.__name__
            if class_name == 'LinearRegression':
                kaiming_init_linear(module)
            elif class_name == 'Conv2D':
                kaiming_init_conv2d(module)
    
    # Load your dataset
    train_dataset = CatAndDogDataset(img_dir='../data/processed', train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    # Run debug analysis
    debug_results = run_comprehensive_debug(model, train_loader, device)
    
    return debug_results
    
    print("Uncomment and modify the debug_your_model() function to run with your specific setup")

if __name__ == "__main__":
    print("Deep Learning Debug Script")
    print("Import this script and call run_comprehensive_debug(model, train_loader) with your model and data")
    debug_your_model()