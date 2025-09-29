"""
Inference optimization for pose estimation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class InferenceOptimizer:
    """Optimizes inference pipeline for pose estimation."""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize inference optimizer.
        
        Args:
            device: Device to optimize for ('auto', 'cpu', 'cuda')
        """
        self.device = self._select_device(device)
        self.optimization_cache = {}
        
        print(f"Inference optimizer initialized for device: {self.device}")
    
    def _select_device(self, device: str) -> str:
        """Select optimal device for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def optimize_inference_pipeline(
        self, 
        model: nn.Module,
        input_shape: Tuple[int, ...],
        optimization_level: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Optimize the entire inference pipeline.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape
            optimization_level: Optimization level ('speed', 'balanced', 'memory')
        
        Returns:
            Dictionary containing optimization results
        """
        print(f"Optimizing inference pipeline for {optimization_level} optimization...")
        
        optimizations = {}
        
        # Model optimization
        optimizations['model'] = self._optimize_model(model, optimization_level)
        
        # Input preprocessing optimization
        optimizations['preprocessing'] = self._optimize_preprocessing(input_shape)
        
        # Postprocessing optimization
        optimizations['postprocessing'] = self._optimize_postprocessing()
        
        # Memory optimization
        optimizations['memory'] = self._optimize_memory_usage(model)
        
        # Batch processing optimization
        optimizations['batching'] = self._optimize_batch_processing()
        
        return optimizations
    
    def _optimize_model(self, model: nn.Module, optimization_level: str) -> Dict[str, Any]:
        """Optimize model for inference."""
        model.eval()
        model = model.to(self.device)
        
        optimizations = {}
        
        # Enable optimizations
        if self.device == 'cuda':
            optimizations['cudnn_benchmark'] = self._enable_cudnn_benchmark()
            optimizations['tensorrt'] = self._check_tensorrt_availability()
        
        # Model compilation
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                optimizations['torch_compile'] = True
            except Exception as e:
                optimizations['torch_compile'] = False
                optimizations['torch_compile_error'] = str(e)
        
        # Precision optimization
        if optimization_level == 'speed' and self.device == 'cuda':
            model = model.half()
            optimizations['half_precision'] = True
        
        optimizations['optimized_model'] = model
        
        return optimizations
    
    def _enable_cudnn_benchmark(self) -> bool:
        """Enable cuDNN benchmark mode."""
        try:
            torch.backends.cudnn.benchmark = True
            return True
        except:
            return False
    
    def _check_tensorrt_availability(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False
    
    def _optimize_preprocessing(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Optimize input preprocessing."""
        optimizations = {}
        
        # Pre-allocate tensors
        batch_size = input_shape[0] if len(input_shape) > 0 else 1
        optimizations['preallocated_tensor'] = torch.zeros(input_shape, device=self.device)
        
        # Normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        optimizations['normalization_mean'] = mean
        optimizations['normalization_std'] = std
        
        return optimizations
    
    def _optimize_postprocessing(self) -> Dict[str, Any]:
        """Optimize output postprocessing."""
        optimizations = {}
        
        # Pre-allocate output tensors
        optimizations['output_cache'] = {}
        
        return optimizations
    
    def _optimize_memory_usage(self, model: nn.Module) -> Dict[str, Any]:
        """Optimize memory usage."""
        optimizations = {}
        
        if self.device == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if needed
            optimizations['memory_cleared'] = True
        
        # Model size
        optimizations['model_size_mb'] = self._get_model_size(model)
        
        return optimizations
    
    def _optimize_batch_processing(self) -> Dict[str, Any]:
        """Optimize batch processing."""
        optimizations = {}
        
        # Optimal batch sizes for different scenarios
        optimizations['optimal_batch_sizes'] = {
            'cpu': [1, 2, 4, 8],
            'cuda': [1, 4, 8, 16, 32]
        }
        
        return optimizations
    
    def benchmark_inference(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Benchmark inference performance."""
        model.eval()
        model = model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.no_grad():
                    _ = model(sample_input)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
                times.append(elapsed_time)
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)
            else:
                start_time = time.time()
                with torch.no_grad():
                    _ = model(sample_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
        
        return {
            'mean_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'fps': 1000 / np.mean(times) if np.mean(times) > 0 else 0,
            'memory_usage_mb': np.mean(memory_usage) if memory_usage else 0
        }
    
    def optimize_batch_size(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        max_batch_size: int = 32
    ) -> int:
        """Find optimal batch size for inference."""
        print("Finding optimal batch size...")
        
        model.eval()
        model = model.to(self.device)
        
        best_batch_size = 1
        best_throughput = 0
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
        
        for batch_size in batch_sizes:
            try:
                # Create batch input
                batch_input = sample_input.repeat(batch_size, 1, 1, 1)
                
                # Benchmark
                benchmark_result = self.benchmark_inference(model, batch_input, num_runs=50)
                
                # Calculate throughput (images per second)
                throughput = batch_size * benchmark_result['fps']
                
                print(f"Batch size {batch_size}: {throughput:.1f} images/sec")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
            
            except RuntimeError as e:
                print(f"Batch size {batch_size} failed: {e}")
                break
        
        print(f"Optimal batch size: {best_batch_size} (throughput: {best_throughput:.1f} images/sec)")
        return best_batch_size
    
    def create_inference_engine(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        engine_type: str = 'torch'
    ):
        """Create optimized inference engine."""
        if engine_type == 'torch':
            return self._create_torch_engine(model, sample_input)
        elif engine_type == 'onnx':
            return self._create_onnx_engine(model, sample_input)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
    
    def _create_torch_engine(self, model: nn.Module, sample_input: torch.Tensor):
        """Create optimized PyTorch inference engine."""
        class OptimizedTorchEngine:
            def __init__(self, model, device, optimizations):
                self.model = model
                self.device = device
                self.optimizations = optimizations
                self.model.eval()
                self.model = self.model.to(device)
                
                # Pre-allocate tensors
                self.input_cache = {}
                self.output_cache = {}
            
            def predict(self, input_tensor):
                input_tensor = input_tensor.to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                return output
            
            def predict_batch(self, input_batch):
                input_batch = input_batch.to(self.device)
                
                with torch.no_grad():
                    output_batch = self.model(input_batch)
                
                return output_batch
        
        return OptimizedTorchEngine(model, self.device, self.optimization_cache)
    
    def _create_onnx_engine(self, model: nn.Module, sample_input: torch.Tensor):
        """Create optimized ONNX inference engine."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        # Export to ONNX
        onnx_path = "temp_model.onnx"
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        class OptimizedONNXEngine:
            def __init__(self, session, input_name, output_name):
                self.session = session
                self.input_name = input_name
                self.output_name = output_name
            
            def predict(self, input_tensor):
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = input_tensor.cpu().numpy()
                
                output = self.session.run([self.output_name], {self.input_name: input_tensor})
                return torch.from_numpy(output[0])
            
            def predict_batch(self, input_batch):
                if isinstance(input_batch, torch.Tensor):
                    input_batch = input_batch.cpu().numpy()
                
                output = self.session.run([self.output_name], {self.input_name: input_batch})
                return torch.from_numpy(output[0])
        
        return OptimizedONNXEngine(session, 'input', 'output')
    
    def optimize_for_latency(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Optimize model specifically for low latency."""
        print("Optimizing for low latency...")
        
        optimizations = {}
        
        # Model optimizations
        model.eval()
        model = model.to(self.device)
        
        # Enable all optimizations
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Use half precision if available
        if self.device == 'cuda':
            model = model.half()
            sample_input = sample_input.half()
            optimizations['half_precision'] = True
        
        # Compile model if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                optimizations['torch_compile'] = True
            except Exception as e:
                optimizations['torch_compile'] = False
        
        # Benchmark optimized model
        benchmark_result = self.benchmark_inference(model, sample_input, num_runs=200)
        optimizations['benchmark'] = benchmark_result
        
        optimizations['optimized_model'] = model
        
        return optimizations
    
    def optimize_for_throughput(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Optimize model specifically for high throughput."""
        print("Optimizing for high throughput...")
        
        optimizations = {}
        
        # Find optimal batch size
        optimal_batch_size = self.optimize_batch_size(model, sample_input)
        optimizations['optimal_batch_size'] = optimal_batch_size
        
        # Create batch input
        batch_input = sample_input.repeat(optimal_batch_size, 1, 1, 1)
        
        # Optimize model
        model.eval()
        model = model.to(self.device)
        
        # Enable optimizations
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Benchmark batch processing
        benchmark_result = self.benchmark_inference(model, batch_input, num_runs=100)
        benchmark_result['throughput_images_per_sec'] = optimal_batch_size * benchmark_result['fps']
        
        optimizations['benchmark'] = benchmark_result
        optimizations['optimized_model'] = model
        
        return optimizations
    
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
    
    def compare_optimization_levels(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Compare different optimization levels."""
        print("Comparing optimization levels...")
        
        results = {}
        
        # Baseline (no optimization)
        baseline_model = copy.deepcopy(model)
        baseline_result = self.benchmark_inference(baseline_model, sample_input)
        results['baseline'] = baseline_result
        
        # Speed optimization
        speed_optimizations = self.optimize_for_latency(model, sample_input)
        results['speed_optimized'] = speed_optimizations['benchmark']
        
        # Throughput optimization
        throughput_optimizations = self.optimize_for_throughput(model, sample_input)
        results['throughput_optimized'] = throughput_optimizations['benchmark']
        
        # Calculate improvements
        baseline_fps = baseline_result['fps']
        
        results['improvements'] = {
            'speed_improvement': (speed_optimizations['benchmark']['fps'] - baseline_fps) / baseline_fps * 100,
            'throughput_improvement': (throughput_optimizations['benchmark']['throughput_images_per_sec'] - baseline_fps) / baseline_fps * 100
        }
        
        return results
