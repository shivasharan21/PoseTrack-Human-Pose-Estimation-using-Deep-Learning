"""
Model quantization for reduced size and faster inference.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List
import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quantization_type: str = 'dynamic'  # 'dynamic', 'static', 'qat'
    dtype: str = 'qint8'  # 'qint8', 'qint16', 'float16'
    calibration_dataset: Optional[str] = None
    num_calibration_samples: int = 100
    target_device: str = 'cpu'  # 'cpu', 'cuda'
    optimize_for: str = 'size'  # 'size', 'speed', 'accuracy'
    export_format: str = 'torch'  # 'torch', 'onnx', 'tensorrt'


class ModelQuantizer:
    """Quantizes models for optimization."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize model quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        self.quantized_model = None
        self.calibration_data = None
    
    def quantize_model(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        calibration_dataset: Optional[List[torch.Tensor]] = None
    ) -> nn.Module:
        """
        Quantize a PyTorch model.
        
        Args:
            model: PyTorch model to quantize
            sample_input: Sample input tensor
            calibration_dataset: Dataset for calibration (for static quantization)
        
        Returns:
            Quantized model
        """
        print(f"Quantizing model using {self.config.quantization_type} quantization...")
        
        # Set model to evaluation mode
        model.eval()
        
        if self.config.quantization_type == 'dynamic':
            return self._dynamic_quantization(model)
        elif self.config.quantization_type == 'static':
            return self._static_quantization(model, sample_input, calibration_dataset)
        elif self.config.quantization_type == 'qat':
            return self._quantization_aware_training(model)
        else:
            raise ValueError(f"Unsupported quantization type: {self.config.quantization_type}")
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        print("Applying dynamic quantization...")
        
        # Quantize the model
        if self.config.dtype == 'qint8':
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif self.config.dtype == 'float16':
            quantized_model = model.half()
        else:
            raise ValueError(f"Unsupported dtype for dynamic quantization: {self.config.dtype}")
        
        print(f"Dynamic quantization completed. Model size: {self._get_model_size(quantized_model):.2f} MB")
        
        return quantized_model
    
    def _static_quantization(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        calibration_dataset: Optional[List[torch.Tensor]] = None
    ) -> nn.Module:
        """Apply static quantization."""
        print("Applying static quantization...")
        
        # Prepare model for quantization
        model.eval()
        
        # Set quantization configuration
        if self.config.dtype == 'qint8':
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            raise ValueError("Static quantization only supports qint8")
        
        # Prepare the model
        prepared_model = torch.quantization.prepare(model)
        
        # Calibrate the model
        if calibration_dataset:
            print("Calibrating model...")
            self._calibrate_model(prepared_model, calibration_dataset)
        else:
            # Use sample input for basic calibration
            with torch.no_grad():
                prepared_model(sample_input)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        print(f"Static quantization completed. Model size: {self._get_model_size(quantized_model):.2f} MB")
        
        return quantized_model
    
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Apply quantization aware training."""
        print("Setting up quantization aware training...")
        
        # Set QAT configuration
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model for QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        print("Model prepared for quantization aware training")
        print("Note: Model needs to be trained with QAT configuration")
        
        return prepared_model
    
    def _calibrate_model(self, model: nn.Module, calibration_dataset: List[torch.Tensor]):
        """Calibrate model for static quantization."""
        model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(calibration_dataset):
                if i >= self.config.num_calibration_samples:
                    break
                
                if isinstance(data, (list, tuple)):
                    model(data[0])
                else:
                    model(data)
                
                if (i + 1) % 10 == 0:
                    print(f"Calibrated {i + 1}/{self.config.num_calibration_samples} samples")
    
    def export_quantized_model(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        output_path: str
    ):
        """Export quantized model in various formats."""
        model.eval()
        
        if self.config.export_format == 'torch':
            self._export_torch_model(model, output_path)
        elif self.config.export_format == 'onnx':
            self._export_onnx_model(model, sample_input, output_path)
        elif self.config.export_format == 'tensorrt':
            self._export_tensorrt_model(model, sample_input, output_path)
        else:
            raise ValueError(f"Unsupported export format: {self.config.export_format}")
    
    def _export_torch_model(self, model: nn.Module, output_path: str):
        """Export model as PyTorch format."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'quantized',
            'quantization_config': self.config
        }, output_path)
        
        print(f"Quantized model saved to: {output_path}")
    
    def _export_onnx_model(self, model: nn.Module, sample_input: torch.Tensor, output_path: str):
        """Export model as ONNX format."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
        
        print("Exporting to ONNX format...")
        
        # Export to ONNX
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
        
        print(f"ONNX model saved to: {output_path}")
        
        # Verify ONNX model
        self._verify_onnx_model(output_path)
    
    def _export_tensorrt_model(self, model: nn.Modch.Tensor, sample_input: torch.Tensor, output_path: str):
        """Export model as TensorRT format."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        print("Exporting to TensorRT format...")
        
        # First export to ONNX
        onnx_path = output_path.replace('.trt', '.onnx')
        self._export_onnx_model(model, sample_input, onnx_path)
        
        # Convert ONNX to TensorRT
        self._convert_onnx_to_tensorrt(onnx_path, output_path)
        
        print(f"TensorRT model saved to: {output_path}")
    
    def _convert_onnx_to_tensorrt(self, onnx_path: str, trt_path: str):
        """Convert ONNX model to TensorRT."""
        # Create TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)
        
        # Create builder
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
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Set optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
    
    def _verify_onnx_model(self, onnx_path: str):
        """Verify ONNX model."""
        if not ONNX_AVAILABLE:
            return
        
        print("Verifying ONNX model...")
        
        # Load and check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("ONNX model verification passed")
    
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
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                    _ = model(sample_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
                else:
                    import time
                    start = time.time()
                    _ = model(sample_input)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'fps': 1000 / np.mean(times) if np.mean(times) > 0 else 0
        }
    
    def compare_models(
        self, 
        original_model: nn.Module, 
        quantized_model: nn.Module,
        sample_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Compare original and quantized models."""
        # Get model sizes
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        # Benchmark models
        original_benchmark = self.benchmark_model(original_model, sample_input)
        quantized_benchmark = self.benchmark_model(quantized_model, sample_input)
        
        # Calculate accuracy (if possible)
        original_model.eval()
        quantized_model.eval()
        
        with torch.no_grad():
            original_output = original_model(sample_input)
            quantized_output = quantized_model(sample_input)
            
            # Calculate MSE between outputs
            mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
        
        return {
            'model_sizes': {
                'original_mb': original_size,
                'quantized_mb': quantized_size,
                'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0,
                'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
            },
            'performance': {
                'original': original_benchmark,
                'quantized': quantized_benchmark,
                'speedup': original_benchmark['mean_inference_time_ms'] / quantized_benchmark['mean_inference_time_ms'] if quantized_benchmark['mean_inference_time_ms'] > 0 else 0
            },
            'accuracy': {
                'mse': mse,
                'relative_error': mse / torch.mean(original_output**2).item() if torch.mean(original_output**2).item() > 0 else 0
            }
        }


class ONNXOptimizer:
    """Optimizes ONNX models for inference."""
    
    def __init__(self):
        """Initialize ONNX optimizer."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
    
    def optimize_onnx_model(self, input_path: str, output_path: str):
        """Optimize ONNX model."""
        print(f"Optimizing ONNX model: {input_path}")
        
        # Load model
        model = onnx.load(input_path)
        
        # Apply optimizations
        optimized_model = self._apply_optimizations(model)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        print(f"Optimized ONNX model saved to: {output_path}")
        
        # Verify optimized model
        onnx.checker.check_model(optimized_model)
        print("Optimized model verification passed")
    
    def _apply_optimizations(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply various optimizations to ONNX model."""
        from onnx import optimizer
        
        # Define optimization passes
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm',
            'eliminate_unused_initializer',
            'extract_constant_to_initializer',
            'eliminate_duplicate_initializer',
            'fuse_add_bias_into_conv',
            'fuse_consecutive_concats'
        ]
        
        # Apply optimizations
        optimized_model = optimizer.optimize(model, passes)
        
        return optimized_model
    
    def create_quantized_onnx(self, input_path: str, output_path: str):
        """Create quantized ONNX model."""
        print(f"Creating quantized ONNX model: {input_path}")
        
        # This would require onnxruntime quantization tools
        # For now, provide a placeholder implementation
        print("ONNX quantization requires onnxruntime quantization tools")
        print("Please use onnxruntime.quantization for ONNX model quantization")
        
        # Copy input to output for now
        import shutil
        shutil.copy(input_path, output_path)
        print(f"Model copied to: {output_path}")
