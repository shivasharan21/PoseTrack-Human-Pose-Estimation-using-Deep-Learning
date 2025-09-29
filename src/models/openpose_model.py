"""
OpenPose-style pose estimation model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
from .base_model import BasePoseModel, PoseHead, MultiScaleFeatureExtractor, AttentionModule


class VGGBlock(nn.Module):
    """VGG-style block for OpenPose backbone."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class OpenPoseBackbone(nn.Module):
    """OpenPose backbone network (VGG-like architecture)."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Stage 1
        self.stage1 = nn.Sequential(
            VGGBlock(in_channels, 64),
            nn.MaxPool2d(2, 2),
            VGGBlock(64, 128),
            nn.MaxPool2d(2, 2),
            VGGBlock(128, 256),
            nn.MaxPool2d(2, 2),
            VGGBlock(256, 512),
            VGGBlock(512, 512)
        )
        
        # Stage 2-6
        self.stages = nn.ModuleList([
            nn.Sequential(
                VGGBlock(512, 512),
                VGGBlock(512, 512),
                VGGBlock(512, 512)
            ) for _ in range(5)
        ])
        
        # Final conv
        self.final_conv = nn.Conv2d(512, 512, 1)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through backbone."""
        features = []
        
        # Stage 1
        x = self.stage1(x)
        features.append(x)
        
        # Stages 2-6
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # Final conv
        x = self.final_conv(x)
        features.append(x)
        
        return features


class PartAffinityField(nn.Module):
    """Part Affinity Field (PAF) module."""
    
    def __init__(self, in_channels: int, num_pafs: int = 38):
        super().__init__()
        self.num_pafs = num_pafs
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_pafs, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class ConfidenceMap(nn.Module):
    """Confidence map module."""
    
    def __init__(self, in_channels: int, num_keypoints: int = 19):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class OpenPoseModel(BasePoseModel):
    """OpenPose-style pose estimation model."""
    
    def __init__(
        self,
        num_keypoints: int = 19,  # OpenPose format
        num_pafs: int = 38,
        input_size: Tuple[int, int] = (368, 368),
        output_stride: int = 8
    ):
        """
        Initialize OpenPose model.
        
        Args:
            num_keypoints: Number of keypoints
            num_pafs: Number of Part Affinity Fields
            input_size: Input image size
            output_stride: Output stride
        """
        super().__init__(num_keypoints, input_size, output_stride)
        
        self.num_pafs = num_pafs
        
        # Backbone
        self.backbone = OpenPoseBackbone()
        
        # Stage 1: Initial predictions
        self.stage1_paf = PartAffinityField(512, num_pafs)
        self.stage1_cm = ConfidenceMap(512, num_keypoints)
        
        # Stages 2-6: Refinement
        self.refinement_stages = nn.ModuleList()
        for i in range(5):
            stage_paf = PartAffinityField(512 + num_pafs + num_keypoints, num_pafs)
            stage_cm = ConfidenceMap(512 + num_pafs + num_keypoints, num_keypoints)
            self.refinement_stages.append(nn.ModuleDict({
                'paf': stage_paf,
                'cm': stage_cm
            }))
        
        # Attention modules for refinement
        self.attention_modules = nn.ModuleList([
            AttentionModule(512 + num_pafs + num_keypoints) for _ in range(5)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through OpenPose model."""
        # Get backbone features
        backbone_features = self.backbone(x)
        features = backbone_features[-1]  # Use final features
        
        # Stage 1 predictions
        stage1_paf = self.stage1_paf(features)
        stage1_cm = self.stage1_cm(features)
        
        outputs = {
            'stage1_paf': stage1_paf,
            'stage1_cm': stage1_cm
        }
        
        # Refinement stages
        current_features = features
        for i, (stage, attention) in enumerate(zip(self.refinement_stages, self.attention_modules)):
            # Apply attention
            attended_features = attention(current_features)
            
            # Concatenate with previous predictions
            stage_input = torch.cat([
                attended_features,
                stage1_paf if i == 0 else outputs[f'stage{i}_paf'],
                stage1_cm if i == 0 else outputs[f'stage{i}_cm']
            ], dim=1)
            
            # Predict PAF and confidence map
            stage_paf = stage['paf'](stage_input)
            stage_cm = stage['cm'](stage_input)
            
            outputs[f'stage{i+2}_paf'] = stage_paf
            outputs[f'stage{i+2}_cm'] = stage_cm
            
            current_features = stage_input
        
        # Final predictions (use last stage)
        outputs['paf'] = outputs['stage6_paf']
        outputs['confidence_map'] = outputs['stage6_cm']
        
        return outputs
    
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
        confidence_map = output['confidence_map'].squeeze(0).cpu().numpy()  # [C, H, W]
        paf = output['paf'].squeeze(0).cpu().numpy()  # [C, H, W]
        
        predictions = []
        
        # Find peaks in confidence maps
        keypoints = []
        for i in range(self.num_keypoints):
            cm = confidence_map[i]
            
            # Find local maxima
            peaks = self._find_peaks(cm, confidence_threshold)
            
            # Convert to image coordinates
            for peak in peaks:
                y, x = peak
                # Scale to original image size
                x_scaled = x * self.output_stride
                y_scaled = y * self.output_stride
                
                keypoints.append({
                    'keypoint_id': i,
                    'x': x_scaled,
                    'y': y_scaled,
                    'confidence': cm[y, x]
                })
        
        # Connect keypoints using PAFs
        skeletons = self._connect_keypoints(keypoints, paf)
        
        predictions.append({
            'keypoints': keypoints,
            'skeletons': skeletons,
            'confidence_map': confidence_map,
            'paf': paf
        })
        
        return predictions
    
    def _find_peaks(self, heatmap: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find peaks in heatmap."""
        from scipy.ndimage import maximum_filter
        
        # Apply non-maximum suppression
        local_maxima = maximum_filter(heatmap, size=3) == heatmap
        
        # Filter by threshold
        peaks = []
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                if local_maxima[y, x] and heatmap[y, x] > threshold:
                    peaks.append((y, x))
        
        return peaks
    
    def _connect_keypoints(
        self, 
        keypoints: List[Dict[str, Any]], 
        paf: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Connect keypoints using Part Affinity Fields."""
        # OpenPose skeleton connections
        skeleton_connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6],
            [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
            [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]
        ]
        
        skeletons = []
        
        for connection in skeleton_connections:
            start_id, end_id = connection
            
            # Find keypoints for this connection
            start_keypoints = [kp for kp in keypoints if kp['keypoint_id'] == start_id]
            end_keypoints = [kp for kp in keypoints if kp['keypoint_id'] == end_id]
            
            if start_keypoints and end_keypoints:
                # Find best connection using PAF
                best_connection = self._find_best_connection(
                    start_keypoints, end_keypoints, paf, connection
                )
                
                if best_connection:
                    skeletons.append(best_connection)
        
        return skeletons
    
    def _find_best_connection(
        self, 
        start_keypoints: List[Dict], 
        end_keypoints: List[Dict], 
        paf: np.ndarray, 
        connection: List[int]
    ) -> Optional[Dict[str, Any]]:
        """Find best connection between two sets of keypoints using PAF."""
        best_score = 0
        best_connection = None
        
        for start_kp in start_keypoints:
            for end_kp in end_keypoints:
                # Calculate PAF score along the connection
                score = self._calculate_paf_score(start_kp, end_kp, paf, connection)
                
                if score > best_score:
                    best_score = score
                    best_connection = {
                        'start_keypoint': start_kp,
                        'end_keypoint': end_kp,
                        'score': score
                    }
        
        return best_connection if best_score > 0.1 else None
    
    def _calculate_paf_score(
        self, 
        start_kp: Dict, 
        end_kp: Dict, 
        paf: np.ndarray, 
        connection: List[int]
    ) -> float:
        """Calculate PAF score between two keypoints."""
        # Get PAF channel for this connection
        paf_channel_x = connection[0] * 2
        paf_channel_y = connection[0] * 2 + 1
        
        # Sample points along the connection
        num_samples = 10
        scores = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(start_kp['x'] * (1 - t) + end_kp['x'] * t)
            y = int(start_kp['y'] * (1 - t) + end_kp['y'] * t)
            
            # Check bounds
            if 0 <= y < paf.shape[1] and 0 <= x < paf.shape[2]:
                # Calculate vector from start to end
                dx = end_kp['x'] - start_kp['x']
                dy = end_kp['y'] - start_kp['y']
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Normalize vector
                    dx /= length
                    dy /= length
                    
                    # Get PAF values
                    paf_x = paf[paf_channel_x, y, x]
                    paf_y = paf[paf_channel_y, y, x]
                    
                    # Calculate dot product
                    score = dx * paf_x + dy * paf_y
                    scores.append(score)
        
        return np.mean(scores) if scores else 0
    
    def get_skeleton_connections(self) -> List[Tuple[int, int]]:
        """Get OpenPose skeleton connections."""
        return [
            [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6],
            [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
            [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenPose model information."""
        info = super().get_model_info()
        info.update({
            'model_type': 'OpenPose',
            'num_pafs': self.num_pafs,
            'num_stages': 6
        })
        return info
