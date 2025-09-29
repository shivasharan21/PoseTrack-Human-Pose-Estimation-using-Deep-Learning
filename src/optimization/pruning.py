"""
Model pruning and compression for pose estimation models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import copy
from dataclasses import dataclass
from torch.nn.utils import prune
import torch.nn.utils.prune as prune_utils


@dataclass
class PruningConfig:
    """Configuration for model pruning."""
    pruning_type: str = 'structured'  # 'structured', 'unstructured', 'global'
    pruning_ratio: float = 0.2  # Fraction of weights to prune
    sparsity_level: float = 0.8  # Target sparsity level
    pruning_criterion: str = 'magnitude'  # 'magnitude', 'gradient', 'random'
    iterative_pruning: bool = True
    fine_tune_after_pruning: bool = True
    preserve_important_layers: bool = True


class ModelPruner:
    """Prunes models to reduce size and improve inference speed."""
    
    def __init__(self, config: PruningConfig):
        """
        Initialize model pruner.
        
        Args:
            config: Pruning configuration
        """
        self.config = config
        self.pruned_model = None
        self.pruning_history = []
    
    def prune_model(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """
        Prune a PyTorch model.
        
        Args:
            model: PyTorch model to prune
            sample_input: Sample input for importance analysis
            importance_scores: Pre-computed importance scores
        
        Returns:
            Pruned model
        """
        print(f"Pruning model using {self.config.pruning_type} pruning...")
        print(f"Target pruning ratio: {self.config.pruning_ratio}")
        
        # Create a copy of the model
        self.pruned_model = copy.deepcopy(model)
        self.pruned_model.eval()
        
        if self.config.pruning_type == 'structured':
            self.pruned_model = self._structured_pruning(self.pruned_model)
        elif self.config.pruning_type == 'unstructured':
            self.pruned_model = self._unstructured_pruning(self.pruned_model, importance_scores)
        elif self.config.pruning_type == 'global':
            self.pruned_model = self._global_pruning(self.pruned_model, importance_scores)
        else:
            raise ValueError(f"Unsupported pruning type: {self.config.pruning_type}")
        
        # Remove pruning reparameterization
        self._remove_pruning_reparameterization()
        
        print(f"Pruning completed. Model size: {self._get_model_size(self.pruned_model):.2f} MB")
        
        return self.pruned_model
    
    def _structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning (removes entire channels/filters)."""
        print("Applying structured pruning...")
        
        # Define layers to prune
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers_to_prune.append((module, 'weight'))
        
        # Apply structured pruning
        prune.global_unstructured(
            layers_to_prune,
            pruning_method=prune.LnStructured,
            amount=self.config.pruning_ratio,
            n=2,  # L2 norm
            dim=0  # Prune along output channels
        )
        
        return model
    
    def _unstructured_pruning(
        self, 
        model: nn.Module, 
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """Apply unstructured pruning (removes individual weights)."""
        print("Applying unstructured pruning...")
        
        # Define pruning method based on criterion
        if self.config.pruning_criterion == 'magnitude':
            pruning_method = prune.L1Unstructured
        elif self.config.pruning_criterion == 'random':
            pruning_method = prune.RandomUnstructured
        else:
            pruning_method = prune.L1Unstructured
        
        # Apply unstructured pruning to each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=self.config.pruning_ratio,
                    n=1,  # L1 norm
                    dim=0
                )
        
        return model
    
    def _global_pruning(
        self, 
        model: nn.Module, 
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """Apply global pruning across all layers."""
        print("Applying global pruning...")
        
        # Collect all parameters
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_ratio
        )
        
        return model
    
    def _remove_pruning_reparameterization(self):
        """Remove pruning reparameterization to make pruning permanent."""
        for name, module in self.pruned_model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    def iterative_pruning(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        target_sparsity: float = 0.8,
        num_iterations: int = 5
    ) -> nn.Module:
        """
        Apply iterative pruning to gradually increase sparsity.
        
        Args:
            model: PyTorch model to prune
            sample_input: Sample input for evaluation
            target_sparsity: Target sparsity level
            num_iterations: Number of pruning iterations
        
        Returns:
            Iteratively pruned model
        """
        print(f"Starting iterative pruning to {target_sparsity} sparsity...")
        
        current_model = copy.deepcopy(model)
        current_sparsity = 0.0
        iteration_sparsity = target_sparsity / num_iterations
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Calculate current pruning amount
            current_pruning_ratio = iteration_sparsity
            
            # Update config
            config = copy.deepcopy(self.config)
            config.pruning_ratio = current_pruning_ratio
            
            # Create new pruner for this iteration
            pruner = ModelPruner(config)
            
            # Prune model
            current_model = pruner.prune_model(current_model, sample_input)
            
            # Calculate actual sparsity
            actual_sparsity = self._calculate_sparsity(current_model)
            current_sparsity = actual_sparsity
            
            # Record pruning step
            self.pruning_history.append({
                'iteration': iteration + 1,
                'target_sparsity': current_sparsity,
                'actual_sparsity': actual_sparsity,
                'model_size_mb': self._get_model_size(current_model)
            })
            
            print(f"Actual sparsity: {actual_sparsity:.3f}")
            
            # Fine-tune if configured
            if self.config.fine_tune_after_pruning:
                print("Fine-tuning pruned model...")
                current_model = self._fine_tune_pruned_model(current_model, sample_input)
        
        self.pruned_model = current_model
        print(f"Iterative pruning completed. Final sparsity: {current_sparsity:.3f}")
        
        return current_model
    
    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity."""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        if total_params == 0:
            return 0.0
        
        return zero_params / total_params
    
    def _fine_tune_pruned_model(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Fine-tune pruned model to recover performance."""
        # This is a placeholder for fine-tuning
        # In practice, you would train the model on your dataset
        
        print("Fine-tuning pruned model...")
        
        # Set to training mode
        model.train()
        
        # Simple fine-tuning loop (placeholder)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Dummy training loop
        for epoch in range(5):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(sample_input)
            
            # Dummy loss (in practice, use real targets)
            dummy_target = torch.zeros_like(output)
            loss = criterion(output, dummy_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Fine-tuning epoch {epoch + 1}, Loss: {loss.item():.4f}")
        
        # Set back to evaluation mode
        model.eval()
        
        return model
    
    def analyze_pruning_impact(
        self, 
        original_model: nn.Module, 
        pruned_model: nn.Module,
        sample_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze the impact of pruning on model performance."""
        print("Analyzing pruning impact...")
        
        # Get model sizes
        original_size = self._get_model_size(original_model)
        pruned_size = self._get_model_size(pruned_model)
        
        # Calculate sparsity
        original_sparsity = self._calculate_sparsity(original_model)
        pruned_sparsity = self._calculate_sparsity(pruned_model)
        
        # Benchmark performance
        original_perf = self._benchmark_model(original_model, sample_input)
        pruned_perf = self._benchmark_model(pruned_model, sample_input)
        
        # Calculate accuracy difference
        accuracy_diff = self._calculate_accuracy_difference(original_model, pruned_model, sample_input)
        
        return {
            'model_sizes': {
                'original_mb': original_size,
                'pruned_mb': pruned_size,
                'size_reduction_percent': (1 - pruned_size / original_size) * 100,
                'compression_ratio': original_size / pruned_size
            },
            'sparsity': {
                'original': original_sparsity,
                'pruned': pruned_sparsity,
                'sparsity_increase': pruned_sparsity - original_sparsity
            },
            'performance': {
                'original': original_perf,
                'pruned': pruned_perf,
                'speedup': original_perf['mean_inference_time_ms'] / pruned_perf['mean_inference_time_ms'] if pruned_perf['mean_inference_time_ms'] > 0 else 0
            },
            'accuracy_impact': {
                'mse_difference': accuracy_diff,
                'relative_error': accuracy_diff / original_perf['mean_inference_time_ms'] if original_perf['mean_inference_time_ms'] > 0 else 0
            }
        }
    
    def _benchmark_model(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """Benchmark model inference performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(sample_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'fps': 1000 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def _calculate_accuracy_difference(
        self, 
        original_model: nn.Module, 
        pruned_model: nn.Module,
        sample_input: torch.Tensor
    ) -> float:
        """Calculate accuracy difference between models."""
        original_model.eval()
        pruned_model.eval()
        
        with torch.no_grad():
            original_output = original_model(sample_input)
            pruned_output = pruned_model(sample_input)
            
            # Calculate MSE difference
            mse = torch.nn.functional.mse_loss(original_output, pruned_output).item()
        
        return mse
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def save_pruned_model(self, model: nn.Module, output_path: str):
        """Save pruned model."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'pruned',
            'pruning_config': self.config,
            'pruning_history': self.pruning_history
        }, output_path)
        
        print(f"Pruned model saved to: {output_path}")
    
    def load_pruned_model(self, model_class: nn.Module, model_path: str) -> nn.Module:
        """Load pruned model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model instance
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load pruning history
        self.pruning_history = checkpoint.get('pruning_history', [])
        
        print(f"Pruned model loaded from: {model_path}")
        return model
    
    def visualize_pruning(self, model: nn.Module, output_path: str = None):
        """Visualize pruning patterns in the model."""
        import matplotlib.pyplot as plt
        
        layer_names = []
        sparsity_levels = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
                sparsity = (weight == 0).sum().item() / weight.numel()
                
                layer_names.append(name)
                sparsity_levels.append(sparsity)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(layer_names)), sparsity_levels)
        
        # Color bars based on sparsity level
        for i, bar in enumerate(bars):
            if sparsity_levels[i] > 0.5:
                bar.set_color('red')
            elif sparsity_levels[i] > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.xlabel('Layer')
        plt.ylabel('Sparsity Level')
        plt.title('Model Pruning Visualization')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(['High Sparsity (>50%)', 'Medium Sparsity (20-50%)', 'Low Sparsity (<20%)'])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Pruning visualization saved to: {output_path}")
        
        plt.show()
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of pruning process."""
        if not self.pruning_history:
            return {}
        
        final_stats = self.pruning_history[-1]
        
        return {
            'total_iterations': len(self.pruning_history),
            'final_sparsity': final_stats['actual_sparsity'],
            'target_sparsity': final_stats['target_sparsity'],
            'final_model_size_mb': final_stats['model_size_mb'],
            'pruning_history': self.pruning_history
        }
