#!/usr/bin/env python3
"""
Enhanced Debug Script for Vanishing Gradients in Deep Learning
Focuses on logit/probability analysis and layer-wise gradient contribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader

# Import your custom modules (assuming they're available)
from dog_and_cat_classifier_cnn_from_scratch.model import ResNet50
from dog_and_cat_classifier_cnn_from_scratch.data import CatAndDogDataset

class EnhancedGradientDebugger:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gradient_norms = defaultdict(list)
        self.weight_norms = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.logit_stats = defaultdict(list)
        self.probability_stats = defaultdict(list)
        self.layer_gradients = OrderedDict()
        self.layer_names = []
        self.hooks = []
        
        # Register hooks for all layers
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks for comprehensive monitoring"""
        def activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.numel() > 0:
                    output_detached = output.detach()
                    
                    # Basic activation stats
                    self.activation_stats[f"{name}_mean"].append(output_detached.mean().cpu().item())
                    self.activation_stats[f"{name}_std"].append(output_detached.std().cpu().item())
                    self.activation_stats[f"{name}_max"].append(output_detached.max().cpu().item())
                    self.activation_stats[f"{name}_min"].append(output_detached.min().cpu().item())
                    
                    # Check for dead neurons (all zeros)
                    zeros = (output_detached == 0).float().mean().cpu().item()
                    self.activation_stats[f"{name}_dead_neurons"].append(zeros)
                    
                    # Check for saturated neurons (near max/min)
                    if 'relu' in name.lower() or 'activation' in name.lower():
                        # For ReLU-like activations
                        saturated = (output_detached >= 6.0).float().mean().cpu().item()
                        self.activation_stats[f"{name}_saturated"].append(saturated)
                    
                    # Check activation distribution health
                    activation_health = self._assess_activation_health(output_detached)
                    for key, value in activation_health.items():
                        self.activation_stats[f"{name}_{key}"].append(value)
                        
            return hook
        
        def gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_tensor = grad_output[0].detach()
                    grad_norm = grad_tensor.norm().cpu().item()
                    grad_mean = grad_tensor.mean().cpu().item()
                    grad_std = grad_tensor.std().cpu().item()
                    
                    # Store detailed gradient information
                    self.gradient_norms[f"{name}_norm"].append(grad_norm)
                    self.gradient_norms[f"{name}_mean"].append(grad_mean)
                    self.gradient_norms[f"{name}_std"].append(grad_std)
                    
                    # Store for layer-wise analysis
                    self.layer_gradients[name] = {
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std,
                        'shape': list(grad_tensor.shape)
                    }
                    
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
    
    def _assess_activation_health(self, activation):
        """Assess the health of activation distributions"""
        activation_flat = activation.flatten()
        
        # Calculate various health metrics
        health_metrics = {}
        
        # Sparsity (percentage of near-zero values)
        near_zero = (torch.abs(activation_flat) < 1e-6).float().mean().item()
        health_metrics['sparsity'] = near_zero
        
        # Variance collapse indicator
        variance = activation_flat.var().item()
        health_metrics['variance'] = variance
        
        # Skewness approximation (third moment)
        if variance > 1e-8:
            mean_val = activation_flat.mean()
            skewness = ((activation_flat - mean_val) ** 3).mean() / (variance ** 1.5)
            health_metrics['skewness'] = skewness.item()
        else:
            health_metrics['skewness'] = 0.0
        
        return health_metrics
    
    def analyze_logits_and_probabilities(self, data_loader, num_batches=5):
        """Comprehensive analysis of model logits and probabilities"""
        print("\n" + "=" * 70)
        print("LOGIT AND PROBABILITY ANALYSIS")
        print("=" * 70)
        
        self.model.eval()
        
        all_logits = []
        all_probabilities = []
        all_max_probs = []
        all_entropy = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                probabilities = F.softmax(logits, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                
                # Calculate entropy (measure of uncertainty)
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                
                # Store results
                all_logits.append(logits.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_max_probs.append(max_probs.cpu().numpy())
                all_entropy.append(entropy.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                print(f"\nBatch {batch_idx + 1}/{num_batches}:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logit statistics:")
                print(f"    Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"    Mean: {logits.mean().item():.4f}")
                print(f"    Std: {logits.std().item():.4f}")
                
                print(f"  Probability statistics:")
                print(f"    Max prob range: [{max_probs.min().item():.4f}, {max_probs.max().item():.4f}]")
                print(f"    Mean max prob: {max_probs.mean().item():.4f}")
                print(f"    Mean entropy: {entropy.mean().item():.4f}")
                
                # Check for problematic patterns
                self._check_logit_problems(logits, probabilities, batch_idx + 1)
        
        # Aggregate analysis
        all_logits = np.concatenate(all_logits, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        all_max_probs = np.concatenate(all_max_probs, axis=0)
        all_entropy = np.concatenate(all_entropy, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\n" + "="*50)
        print(f"AGGREGATE ANALYSIS ({len(all_logits)} samples)")
        print(f"="*50)
        
        # Logit analysis
        print(f"\nLogit Analysis:")
        print(f"  Overall logit range: [{np.min(all_logits):.4f}, {np.max(all_logits):.4f}]")
        print(f"  Logit mean: {np.mean(all_logits):.4f}")
        print(f"  Logit std: {np.std(all_logits):.4f}")
        
        # Check for class-wise logit bias
        for class_idx in range(all_logits.shape[1]):
            class_logits = all_logits[:, class_idx]
            print(f"  Class {class_idx} logits: mean={np.mean(class_logits):.4f}, std={np.std(class_logits):.4f}")
        
        # Probability analysis
        print(f"\nProbability Analysis:")
        print(f"  Max probability range: [{np.min(all_max_probs):.4f}, {np.max(all_max_probs):.4f}]")
        print(f"  Mean max probability: {np.mean(all_max_probs):.4f}")
        print(f"  Mean entropy: {np.mean(all_entropy):.4f}")
        
        # Confidence analysis
        very_confident = np.sum(all_max_probs > 0.9) / len(all_max_probs)
        uncertain = np.sum(all_max_probs < 0.6) / len(all_max_probs)
        print(f"  Very confident predictions (>0.9): {very_confident*100:.1f}%")
        print(f"  Uncertain predictions (<0.6): {uncertain*100:.1f}%")
        
        # Prediction accuracy
        accuracy = np.mean(all_predictions == all_labels)
        print(f"  Accuracy: {accuracy*100:.2f}%")
        
        return {
            'logits': all_logits,
            'probabilities': all_probabilities,
            'max_probs': all_max_probs,
            'entropy': all_entropy,
            'predictions': all_predictions,
            'labels': all_labels,
            'accuracy': accuracy
        }
    
    def _check_logit_problems(self, logits, probabilities, batch_num):
        """Check for common logit-related problems"""
        problems = []
        
        # Check for extreme logits
        if logits.max() > 10:
            problems.append(f"Very large logits (max: {logits.max().item():.2f}) - possible numerical instability")
        
        if logits.min() < -10:
            problems.append(f"Very small logits (min: {logits.min().item():.2f}) - possible numerical instability")
        
        # Check for NaN or infinite values
        if torch.isnan(logits).any():
            problems.append("NaN values detected in logits!")
        
        if torch.isinf(logits).any():
            problems.append("Infinite values detected in logits!")
        
        # Check for uniform predictions (model not learning)
        pred_variance = probabilities.var(dim=0).mean()
        if pred_variance < 0.01:
            problems.append(f"Very low prediction variance ({pred_variance:.6f}) - model may not be learning")
        
        # Check for mode collapse (all predictions same)
        _, predicted = torch.max(probabilities, 1)
        unique_preds = len(torch.unique(predicted))
        if unique_preds == 1:
            problems.append("Mode collapse - all predictions are the same class")
        
        if problems:
            print(f"    ⚠️  Batch {batch_num} Issues:")
            for problem in problems:
                print(f"      - {problem}")
    
    def analyze_layer_gradient_contribution(self, data_loader, criterion, num_batches=3):
        """Analyze which layers contribute most/least to gradients"""
        print("\n" + "=" * 70)
        print("LAYER-WISE GRADIENT CONTRIBUTION ANALYSIS")
        print("=" * 70)
        
        self.model.train()
        
        layer_gradient_history = defaultdict(list)
        
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
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Collect gradient information for each layer
            gradient_info = OrderedDict()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    grad_mean = param.grad.data.mean().item()
                    grad_std = param.grad.data.std().item()
                    param_norm = param.data.norm().item()
                    
                    # Calculate relative gradient magnitude
                    relative_grad = grad_norm / (param_norm + 1e-8)
                    
                    gradient_info[name] = {
                        'grad_norm': grad_norm,
                        'grad_mean': grad_mean,
                        'grad_std': grad_std,
                        'param_norm': param_norm,
                        'relative_grad': relative_grad,
                        'param_shape': list(param.shape),
                        'param_count': param.numel()
                    }
                    
                    layer_gradient_history[name].append({
                        'grad_norm': grad_norm,
                        'relative_grad': relative_grad
                    })
            
            # Analyze gradient flow pattern
            self._analyze_gradient_flow_pattern(gradient_info, batch_idx + 1)
        
        # Summarize gradient flow across all batches
        self._summarize_gradient_flow(layer_gradient_history)
        
        return layer_gradient_history
    
    def _analyze_gradient_flow_pattern(self, gradient_info, batch_num):
        """Analyze gradient flow patterns in a single batch"""
        print(f"  Gradient Flow Analysis:")
        
        # Sort layers by gradient magnitude
        sorted_layers = sorted(gradient_info.items(), 
                             key=lambda x: x[1]['grad_norm'], reverse=True)
        
        print(f"    Top 5 layers by gradient magnitude:")
        for i, (name, info) in enumerate(sorted_layers[:5]):
            print(f"      {i+1}. {name}: {info['grad_norm']:.6f} "
                  f"(relative: {info['relative_grad']:.6f})")
        
        print(f"    Bottom 5 layers by gradient magnitude:")
        for i, (name, info) in enumerate(sorted_layers[-5:]):
            print(f"      {i+1}. {name}: {info['grad_norm']:.6f} "
                  f"(relative: {info['relative_grad']:.6f})")
        
        # Check for vanishing gradient indicators
        vanishing_threshold = 1e-7
        vanishing_layers = [name for name, info in gradient_info.items() 
                           if info['grad_norm'] < vanishing_threshold]
        
        if vanishing_layers:
            print(f"    ⚠️  Layers with vanishing gradients (<{vanishing_threshold}):")
            for layer in vanishing_layers[:10]:  # Show first 10
                print(f"      - {layer}")
            if len(vanishing_layers) > 10:
                print(f"      ... and {len(vanishing_layers) - 10} more")
        
        # Check for exploding gradient indicators
        exploding_threshold = 1e2
        exploding_layers = [name for name, info in gradient_info.items() 
                           if info['grad_norm'] > exploding_threshold]
        
        if exploding_layers:
            print(f"    ⚠️  Layers with exploding gradients (>{exploding_threshold}):")
            for layer in exploding_layers:
                print(f"      - {layer}")
    
    def _summarize_gradient_flow(self, layer_gradient_history):
        """Summarize gradient flow across all analyzed batches"""
        print(f"\n" + "="*50)
        print(f"GRADIENT FLOW SUMMARY")
        print(f"="*50)
        
        # Calculate average gradient norms for each layer
        avg_gradients = {}
        for layer_name, history in layer_gradient_history.items():
            avg_grad_norm = np.mean([h['grad_norm'] for h in history])
            avg_relative_grad = np.mean([h['relative_grad'] for h in history])
            std_grad_norm = np.std([h['grad_norm'] for h in history])
            
            avg_gradients[layer_name] = {
                'avg_grad_norm': avg_grad_norm,
                'std_grad_norm': std_grad_norm,
                'avg_relative_grad': avg_relative_grad
            }
        
        # Find problematic layers
        vanishing_layers = {name: info for name, info in avg_gradients.items() 
                           if info['avg_grad_norm'] < 1e-6}
        
        healthy_layers = {name: info for name, info in avg_gradients.items() 
                         if 1e-6 <= info['avg_grad_norm'] <= 1e1}
        
        exploding_layers = {name: info for name, info in avg_gradients.items() 
                           if info['avg_grad_norm'] > 1e1}
        
        print(f"\nLayer Health Summary:")
        print(f"  Healthy gradient flow: {len(healthy_layers)} layers")
        print(f"  Vanishing gradients: {len(vanishing_layers)} layers")
        print(f"  Exploding gradients: {len(exploding_layers)} layers")
        
        if vanishing_layers:
            print(f"\n  ⚠️  Layers with vanishing gradients:")
            sorted_vanishing = sorted(vanishing_layers.items(), 
                                    key=lambda x: x[1]['avg_grad_norm'])
            for name, info in sorted_vanishing[:10]:
                print(f"    - {name}: avg_norm={info['avg_grad_norm']:.2e}")
        
        if exploding_layers:
            print(f"\n  ⚠️  Layers with exploding gradients:")
            sorted_exploding = sorted(exploding_layers.items(), 
                                    key=lambda x: x[1]['avg_grad_norm'], reverse=True)
            for name, info in sorted_exploding[:5]:
                print(f"    - {name}: avg_norm={info['avg_grad_norm']:.2e}")
        
        return avg_gradients
    
    def check_weight_initialization(self):
        """Enhanced weight initialization analysis"""
        print("=" * 70)
        print("WEIGHT INITIALIZATION ANALYSIS")
        print("=" * 70)
        
        init_stats = {}
        problematic_layers = []
        
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
                
                # Calculate fan_in and fan_out for initialization assessment
                if len(weight_data.shape) >= 2:
                    fan_in = weight_data.shape[-1]
                    fan_out = weight_data.shape[0]
                    
                    # Expected std for different initialization methods
                    xavier_std = np.sqrt(2.0 / (fan_in + fan_out))
                    he_std = np.sqrt(2.0 / fan_in)
                    
                    stats['fan_in'] = fan_in
                    stats['fan_out'] = fan_out
                    stats['xavier_expected_std'] = xavier_std
                    stats['he_expected_std'] = he_std
                    
                    # Check if initialization is reasonable
                    if 'conv' in name.lower():
                        expected_std = he_std
                        init_method = "He (recommended for ReLU)"
                    else:
                        expected_std = xavier_std
                        init_method = "Xavier"
                    
                    std_ratio = stats['std'] / expected_std
                    stats['std_ratio'] = std_ratio
                    stats['expected_method'] = init_method
                
                init_stats[name] = stats
                
                print(f"\n{name}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Mean: {stats['mean']:.6f}")
                print(f"  Std:  {stats['std']:.6f}")
                print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
                
                if 'std_ratio' in stats:
                    print(f"  Expected std ({stats['expected_method']}): {expected_std:.6f}")
                    print(f"  Std ratio (actual/expected): {stats['std_ratio']:.3f}")
                    
                    if stats['std_ratio'] < 0.5:
                        print(f"  ⚠️  WARNING: Under-initialized (std too small)")
                        problematic_layers.append(name)
                    elif stats['std_ratio'] > 2.0:
                        print(f"  ⚠️  WARNING: Over-initialized (std too large)")
                        problematic_layers.append(name)
                    else:
                        print(f"  ✅ Initialization looks reasonable")
                
                # Additional checks
                if abs(stats['mean']) > 0.1:
                    print(f"  ⚠️  WARNING: Non-zero mean initialization")
                    problematic_layers.append(name)
                    
                if stats['std'] < 0.001:
                    print(f"  ⚠️  WARNING: Very small std - possible vanishing gradients")
                    problematic_layers.append(name)
                elif stats['std'] > 1.0:
                    print(f"  ⚠️  WARNING: Very large std - possible exploding gradients")
                    problematic_layers.append(name)
        
        if problematic_layers:
            print(f"\n⚠️  Summary: {len(problematic_layers)} layers may need re-initialization")
        else:
            print(f"\n✅ All weight initializations look reasonable")
        
        return init_stats
    
    def enhanced_recommendations(self, logit_analysis, gradient_history, init_stats):
        """Provide enhanced recommendations based on comprehensive analysis"""
        print("\n" + "=" * 70)
        print("ENHANCED RECOMMENDATIONS")
        print("=" * 70)
        
        recommendations = []
        priority_issues = []
        
        # Analyze logit health
        if logit_analysis:
            logits = logit_analysis['logits']
            probabilities = logit_analysis['probabilities']
            accuracy = logit_analysis['accuracy']
            
            # Check for logit issues
            logit_range = np.max(logits) - np.min(logits)
            logit_std = np.std(logits)
            
            if logit_range > 20:
                priority_issues.append("Extreme logit range detected")
                recommendations.append(
                    "🚨 CRITICAL: Extreme logit values detected. "
                    "Consider gradient clipping and learning rate reduction."
                )
            
            if logit_std < 0.1:
                priority_issues.append("Low logit variance - model not learning")
                recommendations.append(
                    "📚 LEARNING ISSUE: Low logit variance suggests model isn't learning. "
                    "Check learning rate, loss function, and data preprocessing."
                )
            
            # Check prediction confidence
            mean_max_prob = np.mean(logit_analysis['max_probs'])
            if mean_max_prob < 0.6:
                recommendations.append(
                    "🎯 CONFIDENCE: Low prediction confidence. Model may need more training "
                    "or architectural improvements."
                )
            
            if accuracy < 0.6:
                priority_issues.append(f"Low accuracy: {accuracy:.2%}")
                recommendations.append(
                    f"📊 ACCURACY: Low accuracy ({accuracy:.2%}). "
                    "Check data quality, model capacity, and training procedure."
                )
        
        # Analyze gradient flow
        if gradient_history:
            vanishing_count = 0
            exploding_count = 0
            
            for layer_name, history in gradient_history.items():
                avg_grad = np.mean([h['grad_norm'] for h in history])
                if avg_grad < 1e-6:
                    vanishing_count += 1
                elif avg_grad > 1e1:
                    exploding_count += 1
            
            total_layers = len(gradient_history)
            vanishing_percent = (vanishing_count / total_layers) * 100
            exploding_percent = (exploding_count / total_layers) * 100
            
            if vanishing_percent > 20:
                priority_issues.append(f"Vanishing gradients in {vanishing_percent:.1f}% of layers")
                recommendations.append(
                    f"🌊 VANISHING GRADIENTS: {vanishing_percent:.1f}% of layers have vanishing gradients. "
                    "Consider: residual connections, better initialization, batch normalization."
                )
            
            if exploding_percent > 5:
                priority_issues.append(f"Exploding gradients in {exploding_percent:.1f}% of layers")
                recommendations.append(
                    f"💥 EXPLODING GRADIENTS: {exploding_percent:.1f}% of layers have exploding gradients. "
                    "Use gradient clipping with max_norm=1.0."
                )
        
        # Check initialization issues
        if init_stats:
            bad_init_count = 0
            for name, stats in init_stats.items():
                if 'std_ratio' in stats:
                    if stats['std_ratio'] < 0.5 or stats['std_ratio'] > 2.0:
                        bad_init_count += 1
            
            if bad_init_count > 0:
                recommendations.append(
                    f"🎲 INITIALIZATION: {bad_init_count} layers have poor initialization. "
                    "Re-run initialization with proper He/Xavier methods."
                )
        
        # Provide specific fixes
        print("\nPriority Issues:")
        if priority_issues:
            for i, issue in enumerate(priority_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("  ✅ No critical issues detected!")
        
        print(f"\nRecommendations:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  ✅ Model health looks good!")
        
        # Always provide general recommendations
        print(f"\nGeneral Recommendations:")
        print(f"  • Monitor training loss curve for convergence")
        print(f"  • Use learning rate scheduling (reduce on plateau)")
        print(f"  • Consider data augmentation for better generalization")
        print(f"  • Implement early stopping to prevent overfitting")
        print(f"  • Use gradient clipping as a safety measure")
        
        return recommendations, priority_issues
    
    def save_comprehensive_report(self, logit_analysis, gradient_history, init_stats, output_dir='debug_reports'):
        """Save comprehensive debug report with all analyses"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'timestamp': timestamp,
            'logit_analysis': {
                'accuracy': float(logit_analysis['accuracy']) if logit_analysis else None,
                'logit_stats': {
                    'mean': float(np.mean(logit_analysis['logits'])) if logit_analysis else None,
                    'std': float(np.std(logit_analysis['logits'])) if logit_analysis else None,
                    'min': float(np.min(logit_analysis['logits'])) if logit_analysis else None,
                    'max': float(np.max(logit_analysis['logits'])) if logit_analysis else None,
                } if logit_analysis else None
            },
            'gradient_summary': {
                layer: {
                    'avg_grad_norm': float(np.mean([h['grad_norm'] for h in history])),
                    'avg_relative_grad': float(np.mean([h['relative_grad'] for h in history]))
                } for layer, history in gradient_history.items()
            } if gradient_history else None,
            'initialization_stats': init_stats,
            'layer_names': self.layer_names,
            'activation_stats': dict(self.activation_stats),
            'gradient_norms': dict(self.gradient_norms)
        }
        
        report_path = os.path.join(output_dir, f'comprehensive_debug_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Comprehensive debug report saved to: {report_path}")
        return report_path
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def run_enhanced_debug(model, train_loader, device='cuda'):
    """Run enhanced debugging pipeline with logit/probability analysis"""
    print("🔍 Starting Enhanced Deep Learning Debug Analysis")
    print("=" * 80)
    
    # Initialize debugger
    debugger = EnhancedGradientDebugger(model, device)
    
    try:
        # 1. Check weight initialization
        print("Phase 1: Weight Initialization Analysis...")
        init_stats = debugger.check_weight_initialization()
        
        # 2. Analyze logits and probabilities
        print("\nPhase 2: Logit and Probability Analysis...")
        logit_analysis = debugger.analyze_logits_and_probabilities(train_loader, num_batches=5)
        
        # 3. Analyze layer-wise gradient contributions
        print("\nPhase 3: Layer-wise Gradient Analysis...")
        criterion = model.loss if hasattr(model, 'loss') else nn.CrossEntropyLoss()
        gradient_history = debugger.analyze_layer_gradient_contribution(train_loader, criterion, num_batches=3)
        
        # 4. Enhanced recommendations
        print("\nPhase 4: Generating Recommendations...")
        recommendations, priority_issues = debugger.enhanced_recommendations(
            logit_analysis, gradient_history, init_stats
        )
        
        # 5. Save comprehensive report
        report_path = debugger.save_comprehensive_report(
            logit_analysis, gradient_history, init_stats
        )
        
        print("\n" + "=" * 80)
        print("🎯 ENHANCED DEBUG ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Summary of findings
        print(f"\nSUMMARY:")
        print(f"  📊 Analyzed {len(logit_analysis['logits'])} samples")
        print(f"  🏗️  Examined {len(gradient_history)} layers for gradient flow")
        print(f"  ⚖️  Model accuracy: {logit_analysis['accuracy']:.2%}")
        
        if priority_issues:
            print(f"  ⚠️  Found {len(priority_issues)} priority issues")
        else:
            print(f"  ✅ No critical issues detected")
        
        print(f"  📝 Report saved: {report_path}")
        
        return {
            'init_stats': init_stats,
            'logit_analysis': logit_analysis,
            'gradient_history': gradient_history,
            'recommendations': recommendations,
            'priority_issues': priority_issues,
            'report_path': report_path
        }
        
    finally:
        # Clean up hooks
        debugger.cleanup()

def create_gradient_visualization(gradient_history, save_path=None):
    """Create visualization of gradient flow across layers"""
    if not gradient_history:
        print("No gradient history available for visualization")
        return
    
    # Prepare data for plotting
    layer_names = list(gradient_history.keys())
    avg_grad_norms = [np.mean([h['grad_norm'] for h in history]) 
                      for history in gradient_history.values()]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Gradient norms by layer
    x_pos = range(len(layer_names))
    bars = ax1.bar(x_pos, avg_grad_norms)
    
    # Color bars based on gradient magnitude
    for i, (bar, grad_norm) in enumerate(zip(bars, avg_grad_norms)):
        if grad_norm < 1e-6:
            bar.set_color('red')  # Vanishing
        elif grad_norm > 1e1:
            bar.set_color('orange')  # Exploding
        else:
            bar.set_color('green')  # Healthy
    
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Gradient Norm')
    ax1.set_title('Gradient Flow Across Layers')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Healthy (1e-6 to 1e1)'),
        Patch(facecolor='red', label='Vanishing (<1e-6)'),
        Patch(facecolor='orange', label='Exploding (>1e1)')
    ]
    ax1.legend(handles=legend_elements)
    
    # Plot 2: Gradient evolution across batches (for first few layers)
    selected_layers = list(gradient_history.keys())[:min(5, len(gradient_history))]
    for layer_name in selected_layers:
        history = gradient_history[layer_name]
        grad_norms = [h['grad_norm'] for h in history]
        ax2.plot(grad_norms, label=layer_name, marker='o')
    
    ax2.set_xlabel('Batch Number')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Evolution Across Batches (First 5 Layers)')
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Gradient visualization saved to: {save_path}")
    
    plt.show()
    return fig

def create_logit_probability_visualization(logit_analysis, save_path=None):
    """Create visualization of logit and probability distributions"""
    if not logit_analysis:
        print("No logit analysis available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Logit distribution
    logits = logit_analysis['logits'].flatten()
    axes[0, 0].hist(logits, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Logit Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of All Logits')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Max probability distribution
    max_probs = logit_analysis['max_probs']
    axes[0, 1].hist(max_probs, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Max Probabilities')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Entropy distribution
    entropy = logit_analysis['entropy']
    axes[1, 0].hist(entropy, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Entropy Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Class-wise logit comparison
    logits_by_class = logit_analysis['logits']
    for class_idx in range(logits_by_class.shape[1]):
        class_logits = logits_by_class[:, class_idx]
        axes[1, 1].hist(class_logits, bins=30, alpha=0.6, 
                       label=f'Class {class_idx}', edgecolor='black')
    
    axes[1, 1].set_xlabel('Logit Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Logit Distribution by Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Logit and Probability Analysis (Accuracy: {logit_analysis["accuracy"]:.2%})', 
                 fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Logit/Probability visualization saved to: {save_path}")
    
    plt.show()
    return fig

# Quick diagnostic functions
def quick_gradient_check(model, data_loader, device='cuda', num_samples=1):
    """Quick gradient check for immediate diagnosis"""
    print("🚀 Quick Gradient Check")
    print("-" * 40)
    
    model.train()
    criterion = model.loss if hasattr(model, 'loss') else nn.CrossEntropyLoss()
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if batch_idx >= num_samples:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        print(f"Sample {batch_idx + 1}:")
        print(f"  Loss: {loss.item():.6f}")
        
        # Check first few and last few layers
        param_list = list(model.named_parameters())
        
        print("  First 3 layers:")
        for name, param in param_list[:3]:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"    {name}: {grad_norm:.6f}")
                if grad_norm < 1e-7:
                    print(f"      ⚠️  VANISHING GRADIENT!")
        
        print("  Last 3 layers:")
        for name, param in param_list[-3:]:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"    {name}: {grad_norm:.6f}")
                if grad_norm > 1e2:
                    print(f"      ⚠️  EXPLODING GRADIENT!")

def quick_logit_check(model, data_loader, device='cuda', num_samples=1):
    """Quick logit and probability check"""
    print("🎯 Quick Logit Check")
    print("-" * 40)
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= num_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            
            print(f"Sample {batch_idx + 1}:")
            print(f"  Logit range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"  Mean max probability: {max_probs.mean().item():.3f}")
            print(f"  Predictions: {predicted.cpu().numpy()}")
            
            # Quick health checks
            if torch.isnan(logits).any():
                print("  ⚠️  NaN DETECTED in logits!")
            if torch.isinf(logits).any():
                print("  ⚠️  Infinite values DETECTED in logits!")
            if max_probs.mean() < 0.55:
                print("  ⚠️  Very low confidence predictions!")

# Example usage function
def debug_your_model():
    """
    Example of how to use the enhanced debugger with your specific model
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load your model
    try:
        model = ResNet50(num_classes=2, lr=0.1, in_channels=3, dropout_rate=0.3).to(device)
        print("✅ Model loaded successfully")
        
        # Apply Kaiming initialization (your existing code)
        for module in model.modules():
            if hasattr(module, '__class__'):
                class_name = module.__class__.__name__
                if class_name == 'LinearRegression':
                    # Assuming you have these functions defined
                    pass  # nn.kaiming_init_linear(module)
                elif class_name == 'Conv2D':
                    # Assuming you have these functions defined  
                    pass  # nn.kaiming_init_conv2d(module)
        
        # Load your dataset
        train_dataset = CatAndDogDataset(img_dir='../data/processed', train=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        print("✅ Dataset loaded successfully")
        
        # Quick checks first
        print("\n" + "="*60)
        print("QUICK DIAGNOSTIC CHECKS")
        print("="*60)
        
        quick_gradient_check(model, train_loader, device, num_samples=1)
        quick_logit_check(model, train_loader, device, num_samples=1)
        
        # Full analysis
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        debug_results = run_enhanced_debug(model, train_loader, device)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_gradient_visualization(
            debug_results['gradient_history'], 
            save_path='gradient_analysis.png'
        )
        
        create_logit_probability_visualization(
            debug_results['logit_analysis'],
            save_path='logit_probability_analysis.png'
        )
        
        return debug_results
        
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        print("Please ensure your model and dataset are properly configured")
        return None

if __name__ == "__main__":
    print("Enhanced Deep Learning Debug Script")
    print("="*50)
    print("This script provides comprehensive analysis of:")
    print("  • Weight initialization")
    print("  • Logit and probability distributions")
    print("  • Layer-wise gradient flow")
    print("  • Vanishing/exploding gradient detection")
    print("  • Actionable recommendations")
    print("\nUsage:")
    print("  results = run_enhanced_debug(model, train_loader, device)")
    print("  or call debug_your_model() for full pipeline")
    print("="*50)
    
    # Uncomment to run with your model
    # debug_your_model()