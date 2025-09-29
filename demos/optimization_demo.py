"""
Model optimization demo showing quantization, pruning, and GPU acceleration.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import HRNetModel
from src.optimization import ModelQuantizer, QuantizationConfig, GPUOptimizer, ModelPruner, PruningConfig


def main():
    """Main function for optimization demo."""
    parser = argparse.ArgumentParser(description='Model Optimization Demo')
    parser.add_argument('--model', type=str, default='hrnet',
                       help='Model type to optimize')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                       help='Input size (height width)')
    parser.add_argument('--quantization', action='store_true',
                       help='Enable quantization')
    parser.add_argument('--pruning', action='store_true',
                       help='Enable pruning')
    parser.add_argument('--gpu_optimization', action='store_true',
                       help='Enable GPU optimization')
    parser.add_argument('--compare', action='store_true',
                       help='Compare optimized vs original model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run detailed benchmarks')
    
    args = parser.parse_args()
    
    print("Model Optimization Demo")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
    print("=" * 50)
    
    # Create model
    model = HRNetModel(
        num_keypoints=17,
        input_size=tuple(args.input_size)
    )
    
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample input
    sample_input = torch.randn(1, 3, args.input_size[0], args.input_size[1])
    print(f"Sample input shape: {sample_input.shape}")
    
    # Original model benchmark
    print("\n1. Original Model Benchmark")
    print("-" * 30)
    original_model = model.eval()
    original_size = get_model_size(original_model)
    original_time = benchmark_model(original_model, sample_input)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original inference time: {original_time:.3f} ms")
    print(f"Original FPS: {1000/original_time:.1f}")
    
    optimized_model = original_model
    optimization_results = {}
    
    # GPU Optimization
    if args.gpu_optimization:
        print("\n2. GPU Optimization")
        print("-" * 30)
        
        if torch.cuda.is_available():
            gpu_optimizer = GPUOptimizer(device='cuda')
            
            # Optimize model for GPU
            optimized_model = gpu_optimizer.optimize_model_for_gpu(optimized_model)
            sample_input = sample_input.cuda()
            
            # Benchmark GPU performance
            gpu_results = gpu_optimizer.benchmark_gpu_performance(optimized_model, sample_input)
            
            print(f"GPU inference time: {gpu_results['performance']['mean_inference_time_ms']:.3f} ms")
            print(f"GPU FPS: {gpu_results['performance']['fps']:.1f}")
            
            if torch.cuda.is_available():
                print(f"GPU memory usage: {gpu_results['gpu_memory']['mean_usage_mb']:.1f} MB")
            
            optimization_results['gpu'] = gpu_results
        else:
            print("CUDA not available, skipping GPU optimization")
    
    # Quantization
    if args.quantization:
        print("\n3. Model Quantization")
        print("-" * 30)
        
        quant_config = QuantizationConfig(
            quantization_type='dynamic',
            dtype='qint8',
            target_device='cpu'
        )
        
        quantizer = ModelQuantizer(quant_config)
        
        # Quantize model
        quantized_model = quantizer.quantize_model(optimized_model, sample_input)
        
        # Benchmark quantized model
        quant_time = benchmark_model(quantized_model, sample_input)
        quant_size = get_model_size(quantized_model)
        
        print(f"Quantized model size: {quant_size:.2f} MB")
        print(f"Quantized inference time: {quant_time:.3f} ms")
        print(f"Quantized FPS: {1000/quant_time:.1f}")
        
        # Compare with original
        size_reduction = (1 - quant_size/original_size) * 100
        speed_improvement = original_time / quant_time
        
        print(f"Size reduction: {size_reduction:.1f}%")
        print(f"Speed improvement: {speed_improvement:.2f}x")
        
        # Accuracy analysis
        accuracy_comparison = quantizer.compare_models(optimized_model, quantized_model, sample_input)
        print(f"Accuracy impact (MSE): {accuracy_comparison['accuracy']['mse']:.6f}")
        
        optimization_results['quantization'] = {
            'size_mb': quant_size,
            'inference_time_ms': quant_time,
            'fps': 1000/quant_time,
            'size_reduction_percent': size_reduction,
            'speed_improvement': speed_improvement,
            'accuracy_impact': accuracy_comparison['accuracy']['mse']
        }
        
        optimized_model = quantized_model
    
    # Pruning
    if args.pruning:
        print("\n4. Model Pruning")
        print("-" * 30)
        
        prune_config = PruningConfig(
            pruning_type='unstructured',
            pruning_ratio=0.2,
            sparsity_level=0.8
        )
        
        pruner = ModelPruner(prune_config)
        
        # Prune model
        pruned_model = pruner.prune_model(optimized_model, sample_input)
        
        # Benchmark pruned model
        pruned_time = benchmark_model(pruned_model, sample_input)
        pruned_size = get_model_size(pruned_model)
        sparsity = pruner._calculate_sparsity(pruned_model)
        
        print(f"Pruned model size: {pruned_size:.2f} MB")
        print(f"Pruned inference time: {pruned_time:.3f} ms")
        print(f"Pruned FPS: {1000/pruned_time:.1f}")
        print(f"Model sparsity: {sparsity:.3f}")
        
        # Compare with previous model
        size_reduction = (1 - pruned_size/get_model_size(optimized_model)) * 100
        speed_improvement = benchmark_model(optimized_model, sample_input) / pruned_time
        
        print(f"Size reduction: {size_reduction:.1f}%")
        print(f"Speed improvement: {speed_improvement:.2f}x")
        
        optimization_results['pruning'] = {
            'size_mb': pruned_size,
            'inference_time_ms': pruned_time,
            'fps': 1000/pruned_time,
            'sparsity': sparsity,
            'size_reduction_percent': size_reduction,
            'speed_improvement': speed_improvement
        }
        
        optimized_model = pruned_model
    
    # Final comparison
    if args.compare:
        print("\n5. Final Comparison")
        print("-" * 30)
        
        final_size = get_model_size(optimized_model)
        final_time = benchmark_model(optimized_model, sample_input)
        
        total_size_reduction = (1 - final_size/original_size) * 100
        total_speed_improvement = original_time / final_time
        
        print(f"Original model:")
        print(f"  Size: {original_size:.2f} MB")
        print(f"  Inference time: {original_time:.3f} ms")
        print(f"  FPS: {1000/original_time:.1f}")
        
        print(f"Optimized model:")
        print(f"  Size: {final_size:.2f} MB")
        print(f"  Inference time: {final_time:.3f} ms")
        print(f"  FPS: {1000/final_time:.1f}")
        
        print(f"Overall improvements:")
        print(f"  Size reduction: {total_size_reduction:.1f}%")
        print(f"  Speed improvement: {total_speed_improvement:.2f}x")
        
        optimization_results['final_comparison'] = {
            'total_size_reduction_percent': total_size_reduction,
            'total_speed_improvement': total_speed_improvement
        }
    
    # Detailed benchmark
    if args.benchmark:
        print("\n6. Detailed Benchmark")
        print("-" * 30)
        
        # Run multiple iterations for better statistics
        num_runs = 100
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = optimized_model(sample_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        print(f"Detailed benchmark results ({num_runs} runs):")
        print(f"  Mean time: {np.mean(times):.3f} ms")
        print(f"  Std deviation: {np.std(times):.3f} ms")
        print(f"  Min time: {np.min(times):.3f} ms")
        print(f"  Max time: {np.max(times):.3f} ms")
        print(f"  Median time: {np.median(times):.3f} ms")
        print(f"  95th percentile: {np.percentile(times, 95):.3f} ms")
    
    # Save optimization results
    if optimization_results:
        results_path = 'optimization_results.json'
        import json
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, value in optimization_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        serializable_results[key][k] = v.item()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nOptimization results saved to: {results_path}")
    
    print("\nOptimization demo completed!")


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def benchmark_model(model, sample_input, num_runs=50):
    """Benchmark model inference time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(sample_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times)


if __name__ == '__main__':
    main()
