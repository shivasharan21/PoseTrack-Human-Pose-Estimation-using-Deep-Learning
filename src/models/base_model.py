"""
Base model class for pose estimation.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BasePoseModel(nn.Module, ABC):
    """Base class for all pose estimation models."""
    
    def __init__(
        self,
        num_keypoints: int = 17,
        input_size: Tuple[int, int] = (256, 256),
        output_stride: int = 4
    ):
        """
        Initialize base model.
        
        Args:
            num_keypoints: Number of keypoints to detect
            input_size: Input image size (height, width)
            output_stride: Output stride of the model
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.output_stride = output_stride
        self.output_size = (
            input_size[0] // output_stride,
            input_size[1] // output_stride
        )
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def predict_keypoints(
        self, 
        output: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Predict keypoints from model output.
        
        Args:
            output: Model output dictionary
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            List of keypoint predictions
        """
        pass
    
    def preprocess_input(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess input image for the model.
        
        Args:
            image: Input image [H, W, C]
        
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Convert to tensor and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std
        
        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return image
    
    def postprocess_output(
        self, 
        output: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess model output.
        
        Args:
            output: Raw model output
            original_size: Original image size (height, width)
        
        Returns:
            Postprocessed output
        """
        # Scale keypoints back to original image size
        scale_h = original_size[0] / self.input_size[0]
        scale_w = original_size[1] / self.input_size[1]
        
        processed_output = {}
        for key, value in output.items():
            if 'keypoints' in key or 'heatmap' in key:
                # Scale spatial dimensions
                if value.dim() == 4:  # [B, C, H, W]
                    processed_output[key] = torch.nn.functional.interpolate(
                        value, size=original_size, mode='bilinear', align_corners=False
                    )
                elif value.dim() == 3:  # [B, N, 2] for keypoints
                    scaled_value = value.clone()
                    scaled_value[:, :, 0] *= scale_w  # x
                    scaled_value[:, :, 1] *= scale_h  # y
                    processed_output[key] = scaled_value
                else:
                    processed_output[key] = value
            else:
                processed_output[key] = value
        
        return processed_output
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'num_keypoints': self.num_keypoints,
            'input_size': self.input_size,
            'output_stride': self.output_stride,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class PoseHead(nn.Module):
    """Pose estimation head module."""
    
    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        num_layers: int = 3
    ):
        """
        Initialize pose head.
        
        Args:
            in_channels: Number of input channels
            num_keypoints: Number of keypoints to predict
            num_layers: Number of convolution layers
        """
        super().__init__()
        
        layers = []
        channels = in_channels
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ])
            if i < num_layers - 1:
                channels = channels // 2
        
        layers.append(nn.Conv2d(channels, num_keypoints, kernel_size=1))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pose head."""
        return self.head(x)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor for pose estimation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: List[int] = [1, 2, 4, 8]
    ):
        """
        Initialize multi-scale feature extractor.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per scale
            scales: List of dilation scales
        """
        super().__init__()
        
        self.scales = scales
        self.branches = nn.ModuleList()
        
        for scale in scales:
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=3, padding=scale, dilation=scale
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(len(scales) * out_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale extractor."""
        features = []
        for branch in self.branches:
            features.append(branch(x))
        
        # Concatenate features from all scales
        concat_features = torch.cat(features, dim=1)
        
        # Fuse features
        return self.fusion(concat_features)


class AttentionModule(nn.Module):
    """Attention module for pose estimation."""
    
    def __init__(self, in_channels: int):
        """
        Initialize attention module.
        
        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention module."""
        # Channel attention
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # Spatial attention
        sa_input = torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1)
        sa_weight = self.spatial_attention(sa_input)
        x = x * sa_weight
        
        return x
