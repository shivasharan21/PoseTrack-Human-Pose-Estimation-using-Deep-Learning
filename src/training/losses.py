"""
Loss functions for pose estimation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional


class PoseLoss(nn.Module):
    """Combined loss function for pose estimation."""
    
    def __init__(
        self,
        loss_type: str = 'mse',
        keypoint_weight: float = 1.0,
        heatmap_weight: float = 1.0,
        paf_weight: float = 1.0,
        smooth_loss_weight: float = 0.1,
        focal_loss_alpha: float = 2.0,
        focal_loss_gamma: float = 2.0
    ):
        """
        Initialize pose loss function.
        
        Args:
            loss_type: Type of loss function ('mse', 'l1', 'focal', 'wing')
            keypoint_weight: Weight for keypoint loss
            heatmap_weight: Weight for heatmap loss
            paf_weight: Weight for PAF loss
            smooth_loss_weight: Weight for smoothness loss
            focal_loss_alpha: Alpha parameter for focal loss
            focal_loss_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.keypoint_weight = keypoint_weight
        self.heatmap_weight = heatmap_weight
        self.paf_weight = paf_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        
        # Initialize base loss functions
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss()
        elif loss_type == 'focal':
            self.base_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        elif loss_type == 'wing':
            self.base_loss = WingLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Ground truth keypoints [B, N, 3]
        
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Keypoint loss (direct coordinate regression)
        if 'keypoints' in predictions and self.keypoint_weight > 0:
            keypoint_loss = self._calculate_keypoint_loss(predictions['keypoints'], targets)
            total_loss += self.keypoint_weight * keypoint_loss
        
        # Heatmap loss
        if 'heatmap' in predictions and self.heatmap_weight > 0:
            heatmap_loss = self._calculate_heatmap_loss(predictions['heatmap'], targets)
            total_loss += self.heatmap_weight * heatmap_loss
        
        # PAF loss (for OpenPose-style models)
        if 'paf' in predictions and self.paf_weight > 0:
            paf_loss = self._calculate_paf_loss(predictions['paf'], targets)
            total_loss += self.paf_weight * paf_loss
        
        # Smoothness loss
        if self.smooth_loss_weight > 0:
            smooth_loss = self._calculate_smoothness_loss(predictions, targets)
            total_loss += self.smooth_loss_weight * smooth_loss
        
        return total_loss
    
    def _calculate_keypoint_loss(
        self, 
        predicted_keypoints: torch.Tensor, 
        target_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint coordinate loss."""
        # Create visibility mask
        visibility_mask = target_keypoints[:, :, 2] > 0.5  # [B, N]
        
        # Calculate loss only for visible keypoints
        if visibility_mask.sum() > 0:
            pred_coords = predicted_keypoints[visibility_mask]  # [M, 2]
            target_coords = target_keypoints[visibility_mask][:, :2]  # [M, 2]
            
            return self.base_loss(pred_coords, target_coords)
        else:
            return torch.tensor(0.0, device=predicted_keypoints.device)
    
    def _calculate_heatmap_loss(
        self, 
        predicted_heatmaps: torch.Tensor, 
        target_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """Calculate heatmap loss."""
        batch_size, num_keypoints = target_keypoints.shape[:2]
        device = target_keypoints.device
        
        # Generate target heatmaps
        target_heatmaps = self._generate_target_heatmaps(
            target_keypoints, predicted_heatmaps.shape[-2:]
        )
        
        # Calculate loss
        loss = self.base_loss(predicted_heatmaps, target_heatmaps)
        
        return loss
    
    def _calculate_paf_loss(
        self, 
        predicted_pafs: torch.Tensor, 
        target_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Part Affinity Field loss."""
        # Generate target PAFs
        target_pafs = self._generate_target_pafs(
            target_keypoints, predicted_pafs.shape[-2:]
        )
        
        # Calculate loss
        loss = self.base_loss(predicted_pafs, target_pafs)
        
        return loss
    
    def _calculate_smoothness_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate smoothness loss for temporal consistency."""
        smooth_loss = 0.0
        
        # Apply smoothness to heatmaps if available
        if 'heatmap' in predictions:
            heatmaps = predictions['heatmap']
            
            # Calculate spatial smoothness (Laplacian)
            if heatmaps.dim() == 4:  # [B, C, H, W]
                laplacian_kernel = torch.tensor([
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                ], dtype=torch.float32, device=heatmaps.device).view(1, 1, 3, 3)
                
                for i in range(heatmaps.shape[1]):  # For each keypoint
                    heatmap = heatmaps[:, i:i+1, :, :]
                    smooth_penalty = F.conv2d(heatmap, laplacian_kernel, padding=1)
                    smooth_loss += torch.mean(torch.abs(smooth_penalty))
        
        return smooth_loss / heatmaps.shape[1] if heatmaps.shape[1] > 0 else smooth_loss
    
    def _generate_target_heatmaps(
        self, 
        keypoints: torch.Tensor, 
        heatmap_size: tuple
    ) -> torch.Tensor:
        """Generate target heatmaps from keypoints."""
        batch_size, num_keypoints, _ = keypoints.shape
        height, width = heatmap_size
        device = keypoints.device
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        target_heatmaps = torch.zeros(batch_size, num_keypoints, height, width, device=device)
        
        for b in range(batch_size):
            for k in range(num_keypoints):
                kp = keypoints[b, k]
                x, y, visibility = kp[0], kp[1], kp[2]
                
                if visibility > 0.5:
                    # Create Gaussian heatmap
                    sigma = 2.0
                    heatmap = torch.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
                    target_heatmaps[b, k] = heatmap
        
        return target_heatmaps
    
    def _generate_target_pafs(
        self, 
        keypoints: torch.Tensor, 
        paf_size: tuple
    ) -> torch.Tensor:
        """Generate target PAFs from keypoints."""
        # This is a simplified version - in practice, you'd need skeleton connections
        batch_size, num_keypoints, _ = keypoints.shape
        height, width = paf_size
        device = keypoints.device
        
        # For now, return zeros (you'd implement proper PAF generation here)
        num_pafs = 38  # Typical number of PAF channels for OpenPose
        target_pafs = torch.zeros(batch_size, num_pafs, height, width, device=device)
        
        return target_pafs


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss."""
        # Convert targets to probabilities if they're binary
        if targets.max() <= 1.0:
            ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class WingLoss(nn.Module):
    """Wing Loss for robust regression."""
    
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate wing loss."""
        diff = torch.abs(predictions - targets)
        
        # Wing loss formula
        c = self.omega * (1 - np.log(1 + self.omega / self.epsilon))
        
        wing_loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - c
        )
        
        return wing_loss.mean()


class AdaptiveLoss(nn.Module):
    """Adaptive loss that combines multiple loss functions."""
    
    def __init__(self, num_losses: int = 3):
        super().__init__()
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses: list) -> torch.Tensor:
        """Calculate adaptive loss."""
        total_loss = 0.0
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video sequences."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self, 
        current_keypoints: torch.Tensor, 
        previous_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """Calculate temporal consistency loss."""
        if previous_keypoints is None:
            return torch.tensor(0.0, device=current_keypoints.device)
        
        # Calculate velocity
        velocity = current_keypoints - previous_keypoints
        
        # Penalize large changes in velocity (acceleration)
        acceleration = velocity[1:] - velocity[:-1]
        
        # Calculate smoothness penalty
        smoothness_penalty = torch.mean(torch.norm(acceleration, dim=-1))
        
        return self.weight * smoothness_penalty


class MultiScaleLoss(nn.Module):
    """Multi-scale loss for different resolution outputs."""
    
    def __init__(self, scales: list = [1.0, 0.5, 0.25], weights: list = None):
        super().__init__()
        self.scales = scales
        self.weights = weights or [1.0] * len(scales)
        
        if len(self.weights) != len(self.scales):
            raise ValueError("Number of weights must match number of scales")
    
    def forward(self, predictions: list, targets: torch.Tensor) -> torch.Tensor:
        """Calculate multi-scale loss."""
        total_loss = 0.0
        
        for i, (pred, scale, weight) in enumerate(zip(predictions, self.scales, self.weights)):
            # Resize target to match prediction scale
            if scale != 1.0:
                target_size = (int(targets.shape[-2] * scale), int(targets.shape[-1] * scale))
                scaled_target = F.interpolate(targets, size=target_size, mode='bilinear')
            else:
                scaled_target = targets
            
            # Calculate loss for this scale
            scale_loss = F.mse_loss(pred, scaled_target)
            total_loss += weight * scale_loss
        
        return total_loss
