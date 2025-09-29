"""
GPU acceleration and optimization for pose estimation models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Union
import time
import psutil
import GPUtil

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class GPUOptimizer:
    """Optimizes models for GPU acceleration."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize GPU optimizer.
        
        Args:
            device: Device to optimize for ('cuda', 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.device_info = self._get_device_info()
        
        print(f"GPU Optimizer initialized for device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_reserved': torch.cuda.memory_reserved(),
                'gpu_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device()
            })
        
        return info
    
    def optimize_model_for_gpu(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for GPU inference.
        
        Args:
            model: PyTorch model to optimize
        
        Returns:
            Optimized model
        """
        print("Optimizing model for GPU...")
        
        # Move model to GPU
        model = model.to(self.device)
        
        # Apply GPU-specific optimizations
        model = self._apply_gpu_optimizations(model)
        
        # Set to evaluation mode
        model.eval()
        
        # Enable optimizations
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
        
        if hasattr(torch.backends.cudnn, 'deterministic'):
            torch.backends.cudnn.deterministic = False
        
        print("GPU optimization completed")
        return model
    
    def _apply_gpu_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply GPU-specific optimizations."""
        # Convert to half precision if supported
        if self.device == 'cuda':
            # Check if GPU supports FP16
            gpu_capability = torch.cuda.get_device_capability()
            if gpu_capability[0] >= 6:  # Pascal architecture or newer
                print("Converting model to half precision (FP16)")
                model = model.half()
            else:
                print("GPU does not support FP16, using FP32")
        
        # Compile model if PyTorch 2.0+ is available
        if hasattr(torch, 'compile'):
            try:
                print("Compiling model with torch.compile")
                model = torch.compile(model)
            except Exception as e:
                print(f"torch.compile failed: {e}")
        
        return model
    
    def optimize_data_loading(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Optimize data loading for GPU."""
        print("Optimizing data loading...")
        
        # Set optimal number of workers
        num_workers = min(4, psutil.cpu_count())
        
        # Create optimized dataloader
        optimized_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        print(f"Data loading optimized with {num_workers} workers")
        return optimized_loader
    
    def benchmark_gpu_performance(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Benchmark GPU performance."""
        print("Benchmarking GPU performance...")
        
        model.eval()
        model = model.to(self.device)
        
        # Convert input to appropriate device and dtype
        sample_input = sample_input.to(self.device)
        if model.dtype == torch.float16:
            sample_input = sample_input.half()
        
        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running {num_runs} inference iterations...")
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            if self.device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.no_grad():
                    _ = model(sample_input)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
                times.append(elapsed_time)
                
                # Record memory usage
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            else:
                start_time = time.time()
                with torch.no_grad():
                    _ = model(sample_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        results = {
            'device': self.device,
            'model_dtype': str(model.dtype),
            'input_shape': list(sample_input.shape),
            'performance': {
                'mean_inference_time_ms': np.mean(times),
                'std_inference_time_ms': np.std(times),
                'min_inference_time_ms': np.min(times),
                'max_inference_time_ms': np.max(times),
                'fps': 1000 / np.mean(times) if np.mean(times) > 0 else 0
            }
        }
        
        if self.device == 'cuda':
            results.update({
                'gpu_memory': {
                    'mean_usage_mb': np.mean(memory_usage),
                    'max_usage_mb': np.max(memory_usage),
                    'current_usage_mb': torch.cuda.memory_allocated() / 1024**2,
                    'total_available_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2
                },
                'gpu_utilization': self._get_gpu_utilization()
            })
        
        return results
    
    def _get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization information."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'gpu_load_percent': gpu.load * 100,
                    'gpu_memory_used_percent': gpu.memoryUsed / gpu.memoryTotal * 100,
                    'gpu_temperature': gpu.temperature
                }
        except:
            pass
        
        return {}
    
    def optimize_batch_processing(
        self, 
        model: nn.Module, 
        sample_inputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Optimize batch processing for multiple inputs."""
        print("Optimizing batch processing...")
        
        model.eval()
        model = model.to(self.device)
        
        # Stack inputs into batch
        batch_input = torch.stack(sample_inputs).to(self.device)
        
        if model.dtype == torch.float16:
            batch_input = batch_input.half()
        
        # Process batch
        with torch.no_grad():
            batch_output = model(batch_input)
        
        return batch_output
    
    def create_tensorrt_engine(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        engine_path: str,
        max_batch_size: int = 1
    ):
        """Create TensorRT engine for optimized inference."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        print("Creating TensorRT engine...")
        
        # First convert to ONNX
        onnx_path = engine_path.replace('.trt', '.onnx')
        self._export_to_onnx(model, sample_input, onnx_path)
        
        # Create TensorRT engine
        self._build_tensorrt_engine(onnx_path, engine_path, max_batch_size)
        
        print(f"TensorRT engine saved to: {engine_path}")
    
    def _export_to_onnx(self, model: nn.Module, sample_input: torch.Tensor, output_path: str):
        """Export model to ONNX format."""
        model.eval()
        model = model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        if model.dtype == torch.float16:
            sample_input = sample_input.half()
        
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
    def _build_tensorrt_engine(self, onnx_path: str, engine_path: str, max_batch_size: int):
        """Build TensorRT engine from ONNX model."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Create optimization profile
        profile = builder.create_optimization_profile()
        
        # Set input shape range
        input_shape = (1, 3, 224, 224)  # Default input shape
        profile.set_shape("input", (1, *input_shape[1:]), (max_batch_size, *input_shape[1:]), (max_batch_size, *input_shape[1:]))
        config.add_optimization_profile(profile)
        
        # Enable FP16 if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("TensorRT FP16 optimization enabled")
        
        # Build engine
        print("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
    
    def optimize_memory_usage(self, model: nn.Module):
        """Optimize GPU memory usage."""
        print("Optimizing GPU memory usage...")
        
        if self.device == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory fraction if needed
            # torch.cuda.set_per_process_memory_fraction(0.8)
        
        print("Memory optimization completed")
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status."""
        status = {
            'device_info': self.device_info,
            'current_time': time.time()
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            status.update({
                'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'memory_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'memory_cached_mb': torch.cuda.memory_cached() / 1024**2,
                'gpu_utilization': self._get_gpu_utilization()
            })
        
        return status
    
    def profile_model(self, model: nn.Module, sample_input: torch.Tensor):
        """Profile model execution."""
        print("Profiling model execution...")
        
        model.eval()
        model = model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        if model.dtype == torch.float16:
            sample_input = sample_input.half()
        
        # Create profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                _ = model(sample_input)
        
        # Print results
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        return prof
    
    def compare_cpu_gpu_performance(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """Compare CPU vs GPU performance."""
        print("Comparing CPU vs GPU performance...")
        
        results = {}
        
        # CPU performance
        print("Benchmarking CPU performance...")
        cpu_model = model.cpu()
        cpu_input = sample_input.cpu()
        
        cpu_times = []
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = cpu_model(cpu_input)
            
            for _ in range(num_runs):
                start = time.time()
                _ = cpu_model(cpu_input)
                end = time.time()
                cpu_times.append((end - start) * 1000)
        
        # GPU performance
        print("Benchmarking GPU performance...")
        gpu_model = model.cuda()
        gpu_input = sample_input.cuda()
        
        gpu_times = []
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = gpu_model(gpu_input)
            
            torch.cuda.synchronize()
            for _ in range(num_runs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = gpu_model(gpu_input)
                end_event.record()
                
                torch.cuda.synchronize()
                gpu_times.append(start_event.elapsed_time(end_event))
        
        # Calculate results
        results = {
            'cpu': {
                'mean_time_ms': np.mean(cpu_times),
                'std_time_ms': np.std(cpu_times),
                'fps': 1000 / np.mean(cpu_times) if np.mean(cpu_times) > 0 else 0
            },
            'gpu': {
                'mean_time_ms': np.mean(gpu_times),
                'std_time_ms': np.std(gpu_times),
                'fps': 1000 / np.mean(gpu_times) if np.mean(gpu_times) > 0 else 0
            },
            'speedup': np.mean(cpu_times) / np.mean(gpu_times) if np.mean(gpu_times) > 0 else 0
        }
        
        print(f"CPU: {results['cpu']['mean_time_ms']:.2f}ms ({results['cpu']['fps']:.1f} FPS)")
        print(f"GPU: {results['gpu']['mean_time_ms']:.2f}ms ({results['gpu']['fps']:.1f} FPS)")
        print(f"Speedup: {results['speedup']:.2f}x")
        
        return results
