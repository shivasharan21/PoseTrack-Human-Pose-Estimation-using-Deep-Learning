"""
HRNet (High-Resolution Network) implementation for pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .base_model import BasePoseModel


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for HRNet."""
    
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """High-Resolution Module for HRNet."""
    
    def __init__(
        self,
        num_branches: int,
        block: nn.Module,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: str = 'SUM'
    ):
        super().__init__()
        
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        
        # Build branches
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = self._make_one_branch(
                block, num_blocks[i], num_inchannels[i], num_channels[i]
            )
            self.branches.append(branch)
        
        # Build fusion layers
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)
                elif j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                            nn.BatchNorm2d(num_channels[i])
                        )
                    )
                else:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(scale_factor=2**(i-j), mode='nearest')
                        )
                    )
            self.fuse_layers.append(fuse_layer)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _make_one_branch(self, block, num_blocks, num_inchannels, num_channels):
        layers = []
        layers.append(block(num_inchannels, num_channels))
        
        for _ in range(1, num_blocks):
            layers.append(block(num_channels, num_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # Forward through branches
        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])
        
        # Fuse branches
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y += self.fuse_layers[i][j](x[j])
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        
        return x_fuse


class HRNetModel(BasePoseModel):
    """HRNet model for pose estimation."""
    
    def __init__(
        self,
        num_keypoints: int = 17,
        input_size: Tuple[int, int] = (256, 256),
        output_stride: int = 4,
        width: int = 18,  # HRNet-W18
        num_stages: int = 4
    ):
        """
        Initialize HRNet model.
        
        Args:
            num_keypoints: Number of keypoints
            input_size: Input image size
            output_stride: Output stride
            width: HRNet width multiplier
            num_stages: Number of HRNet stages
        """
        super().__init__(num_keypoints, input_size, output_stride)
        
        self.width = width
        self.num_stages = num_stages
        
        # Stage 1: Stem
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 2: First resolution
        self.layer1 = self._make_layer(Bottleneck, 64, 4, 64)
        
        # Stage 3: Two resolutions
        num_channels = [width * 2, width * 4]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            HighResolutionModule, num_channels, 1, BasicBlock, [2, 2]
        )
        
        # Stage 4: Three resolutions
        num_channels = [width * 2, width * 4, width * 8]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            HighResolutionModule, num_channels, 4, BasicBlock, [2, 2, 2]
        )
        
        # Stage 5: Four resolutions
        num_channels = [width * 2, width * 4, width * 8, width * 16]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            HighResolutionModule, num_channels, 3, BasicBlock, [2, 2, 2, 2]
        )
        
        # Final layers
        self.final_layers = nn.ModuleList()
        for i in range(len(pre_stage_channels)):
            final_layer = nn.Sequential(
                nn.Conv2d(pre_stage_channels[i], pre_stage_channels[i], 3, 1, 1),
                nn.BatchNorm2d(pre_stage_channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(pre_stage_channels[i], num_keypoints, 1)
            )
            self.final_layers.append(final_layer)
        
        # Final fusion
        self.final_conv = nn.Conv2d(sum(pre_stage_channels), num_keypoints, 1)
    
    def _make_layer(self, block, in_channels, blocks, out_channels):
        layers = []
        layers.append(block(in_channels, out_channels))
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[-1], num_channels_cur_layer[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)
                    )
                )
        
        return nn.ModuleList(transition_layers)
    
    def _make_stage(self, block, num_channels, num_blocks, block_type, num_inchannels):
        layers = []
        for i in range(num_blocks):
            layers.append(
                block(
                    num_branches=len(num_channels),
                    block=block_type,
                    num_blocks=num_inchannels,
                    num_inchannels=num_channels,
                    num_channels=num_channels
                )
            )
        
        return nn.Sequential(*layers), num_channels
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through HRNet."""
        # Stage 1: Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 2
        x = self.layer1(x)
        
        # Stage 3
        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Stage 4
        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Stage 5
        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Final layers
        final_outputs = []
        for i, final_layer in enumerate(self.final_layers):
            final_outputs.append(final_layer(y_list[i]))
        
        # Upsample all outputs to the same size
        target_size = final_outputs[0].shape[-2:]
        for i in range(len(final_outputs)):
            if final_outputs[i].shape[-2:] != target_size:
                final_outputs[i] = F.interpolate(
                    final_outputs[i], size=target_size, mode='bilinear', align_corners=False
                )
        
        # Final fusion
        fused_features = torch.cat(final_outputs, dim=1)
        final_heatmap = self.final_conv(fused_features)
        
        return {
            'heatmap': final_heatmap,
            'multi_scale_features': y_list
        }
    
    def predict_keypoints(
        self, 
        output: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Predict keypoints from HRNet output.
        
        Args:
            output: Model output dictionary
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            List of keypoint predictions
        """
        heatmap = output['heatmap'].squeeze(0).cpu().numpy()  # [C, H, W]
        
        predictions = []
        keypoints = []
        
        # Find peaks in each heatmap
        for i in range(self.num_keypoints):
            hm = heatmap[i]
            
            # Find global maximum
            max_val = np.max(hm)
            if max_val > confidence_threshold:
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                y, x = max_idx
                
                # Convert to original image coordinates
                x_scaled = x * self.output_stride
                y_scaled = y * self.output_stride
                
                keypoints.append({
                    'keypoint_id': i,
                    'x': x_scaled,
                    'y': y_scaled,
                    'confidence': max_val
                })
            else:
                keypoints.append({
                    'keypoint_id': i,
                    'x': 0,
                    'y': 0,
                    'confidence': 0.0
                })
        
        predictions.append({
            'keypoints': keypoints,
            'heatmap': heatmap
        })
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get HRNet model information."""
        info = super().get_model_info()
        info.update({
            'model_type': 'HRNet',
            'width': self.width,
            'num_stages': self.num_stages
        })
        return info
