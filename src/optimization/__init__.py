"""
Model optimization modules for pose estimation.

This module contains optimization techniques including:
- Model quantization for reduced size and faster inference
- GPU acceleration and optimization
- Model pruning and compression
- Inference optimization
"""

from .quantization import ModelQuantizer, QuantizationConfig
from .gpu_optimization import GPUOptimizer
from .pruning import ModelPruner, PruningConfig
from .inference_optimization import InferenceOptimizer

__all__ = [
    'ModelQuantizer', 'QuantizationConfig',
    'GPUOptimizer',
    'ModelPruner', 'PruningConfig',
    'InferenceOptimizer'
]
