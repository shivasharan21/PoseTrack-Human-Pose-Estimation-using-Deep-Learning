"""
Evaluation metrics for pose estimation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class PoseMetrics:
    """Metrics calculator for pose estimation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics_history = []
    
    def calculate_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for pose estimation.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Ground truth keypoints [B, N, 3]
        
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(predictions, targets))
        
        # PCK metrics
        metrics.update(self._calculate_pck_metrics(predictions, targets))
        
        # MPJPE
        metrics.update(self._calculate_mpjpe(predictions, targets))
        
        # OKS (Object Keypoint Similarity)
        metrics.update(self._calculate_oks_metrics(predictions, targets))
        
        # Per-keypoint metrics
        metrics.update(self._calculate_per_keypoint_metrics(predictions, targets))
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_basic_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate basic metrics."""
        metrics = {}
        
        # Extract predicted keypoints
        if 'keypoints' in predictions:
            pred_keypoints = predictions['keypoints']
            
            # Calculate mean absolute error
            mae = torch.mean(torch.abs(pred_keypoints - targets[:, :, :2]))
            metrics['mae'] = mae.item()
            
            # Calculate root mean square error
            mse = torch.mean((pred_keypoints - targets[:, :, :2]) ** 2)
            metrics['rmse'] = torch.sqrt(mse).item()
            
            # Calculate mean error
            mean_error = torch.mean(torch.norm(pred_keypoints - targets[:, :, :2], dim=-1))
            metrics['mean_error'] = mean_error.item()
        
        return metrics
    
    def _calculate_pck_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor,
        thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> Dict[str, float]:
        """Calculate PCK (Percentage of Correct Keypoints) metrics."""
        metrics = {}
        
        if 'keypoints' not in predictions:
            return metrics
        
        pred_keypoints = predictions['keypoints']
        visibility_mask = targets[:, :, 2] > 0.5
        
        for threshold in thresholds:
            # Calculate distances
            distances = torch.norm(pred_keypoints - targets[:, :, :2], dim=-1)
            
            # Apply visibility mask
            visible_distances = distances[visibility_mask]
            
            if len(visible_distances) > 0:
                # Calculate PCK
                correct_keypoints = (visible_distances <= threshold).float()
                pck = torch.mean(correct_keypoints).item()
                metrics[f'pck_{threshold}'] = pck
            else:
                metrics[f'pck_{threshold}'] = 0.0
        
        return metrics
    
    def _calculate_mpjpe(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate MPJPE (Mean Per Joint Position Error)."""
        metrics = {}
        
        if 'keypoints' not in predictions:
            return metrics
        
        pred_keypoints = predictions['keypoints']
        visibility_mask = targets[:, :, 2] > 0.5
        
        # Calculate per-joint errors
        joint_errors = torch.norm(pred_keypoints - targets[:, :, :2], dim=-1)
        
        # Apply visibility mask
        visible_errors = joint_errors[visibility_mask]
        
        if len(visible_errors) > 0:
            # Calculate mean per joint position error
            mpjpe = torch.mean(visible_errors).item()
            metrics['mpjpe'] = mpjpe
            
            # Calculate median per joint position error
            median_pjpe = torch.median(visible_errors).item()
            metrics['median_pjpe'] = median_pjpe
        
        return metrics
    
    def _calculate_oks_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor,
        sigmas: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Calculate OKS (Object Keypoint Similarity) metrics."""
        metrics = {}
        
        if 'keypoints' not in predictions:
            return metrics
        
        # Default sigmas for COCO keypoints
        if sigmas is None:
            sigmas = [
                0.026, 0.025, 0.025, 0.035, 0.035,  # head
                0.079, 0.079, 0.072, 0.072,          # torso
                0.062, 0.062, 0.107, 0.107,          # arms
                0.087, 0.087, 0.089, 0.089           # legs
            ]
        
        pred_keypoints = predictions['keypoints']
        batch_size, num_keypoints = pred_keypoints.shape[:2]
        
        # Calculate object scale (use head size as reference)
        head_size = torch.norm(targets[:, 5, :2] - targets[:, 6, :2], dim=-1)  # shoulder distance
        object_scale = head_size * 0.6  # Approximate object scale
        
        # Calculate OKS for each sample
        oks_scores = []
        
        for b in range(batch_size):
            sample_oks = []
            
            for k in range(num_keypoints):
                if targets[b, k, 2] > 0.5:  # Visible keypoint
                    # Calculate distance
                    distance = torch.norm(pred_keypoints[b, k] - targets[b, k, :2])
                    
                    # Calculate OKS
                    sigma = sigmas[k] if k < len(sigmas) else 0.079  # default
                    scale = object_scale[b]
                    
                    oks = torch.exp(-distance**2 / (2 * (sigma * scale)**2))
                    sample_oks.append(oks.item())
            
            if sample_oks:
                avg_oks = np.mean(sample_oks)
                oks_scores.append(avg_oks)
        
        if oks_scores:
            metrics['oks_mean'] = np.mean(oks_scores)
            metrics['oks_std'] = np.std(oks_scores)
        
        return metrics
    
    def _calculate_per_keypoint_metrics(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate per-keypoint metrics."""
        metrics = {}
        
        if 'keypoints' not in predictions:
            return metrics
        
        pred_keypoints = predictions['keypoints']
        num_keypoints = pred_keypoints.shape[1]
        
        # Keypoint names (COCO format)
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for k in range(min(num_keypoints, len(keypoint_names))):
            # Get visibility mask for this keypoint
            visibility_mask = targets[:, k, 2] > 0.5
            
            if visibility_mask.sum() > 0:
                # Calculate errors for visible keypoints
                pred_kp = pred_keypoints[:, k]
                target_kp = targets[:, k, :2]
                
                visible_pred = pred_kp[visibility_mask]
                visible_target = target_kp[visibility_mask]
                
                # Calculate mean error
                kp_error = torch.mean(torch.norm(visible_pred - visible_target, dim=-1))
                metrics[f'{keypoint_names[k]}_error'] = kp_error.item()
                
                # Calculate accuracy at different thresholds
                for threshold in [0.1, 0.2, 0.3]:
                    distances = torch.norm(visible_pred - visible_target, dim=-1)
                    accuracy = (distances <= threshold).float().mean()
                    metrics[f'{keypoint_names[k]}_acc_{threshold}'] = accuracy.item()
        
        return metrics
    
    def calculate_heatmap_metrics(
        self, 
        predicted_heatmaps: torch.Tensor, 
        target_heatmaps: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate metrics for heatmap predictions."""
        metrics = {}
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = torch.mean((predicted_heatmaps - target_heatmaps) ** 2)
        if mse > 0:
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            metrics['psnr'] = psnr.item()
        
        # Structural Similarity Index (SSIM) - simplified version
        metrics['ssim'] = self._calculate_ssim(predicted_heatmaps, target_heatmaps)
        
        # Mean Squared Error
        metrics['mse'] = mse.item()
        
        # Mean Absolute Error
        mae = torch.mean(torch.abs(predicted_heatmaps - target_heatmaps))
        metrics['mae'] = mae.item()
        
        return metrics
    
    def _calculate_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate simplified SSIM."""
        # Convert to grayscale if needed
        if x.dim() == 4:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)
        
        # Calculate means
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        
        # Calculate variances and covariance
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Calculate SSIM
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim.item()
    
    def calculate_temporal_metrics(
        self, 
        keypoint_sequence: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Calculate temporal consistency metrics."""
        metrics = {}
        
        if len(keypoint_sequence) < 2:
            return metrics
        
        # Calculate velocity consistency
        velocities = []
        for i in range(1, len(keypoint_sequence)):
            velocity = keypoint_sequence[i] - keypoint_sequence[i-1]
            velocities.append(velocity)
        
        if velocities:
            # Calculate velocity variance (lower is more consistent)
            velocity_tensor = torch.stack(velocities)
            velocity_var = torch.var(velocity_tensor, dim=0)
            metrics['velocity_consistency'] = -torch.mean(velocity_var).item()  # Negative for consistency
            
            # Calculate acceleration
            accelerations = []
            for i in range(1, len(velocities)):
                acceleration = velocities[i] - velocities[i-1]
                accelerations.append(acceleration)
            
            if accelerations:
                acceleration_tensor = torch.stack(accelerations)
                acceleration_magnitude = torch.norm(acceleration_tensor, dim=-1)
                metrics['acceleration_magnitude'] = torch.mean(acceleration_magnitude).item()
        
        return metrics
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get average metrics over history."""
        if not self.metrics_history:
            return {}
        
        history = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        # Calculate averages
        avg_metrics = {}
        for key in history[0].keys():
            values = [metrics[key] for metrics in history if key in metrics]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics over history."""
        if not self.metrics_history:
            return {}
        
        # Find best values for each metric
        best_metrics = {}
        for key in self.metrics_history[0].keys():
            values = [metrics[key] for metrics in self.metrics_history if key in metrics]
            if values:
                # For error metrics, take minimum; for accuracy metrics, take maximum
                if 'error' in key or 'mpjpe' in key or 'mae' in key or 'rmse' in key:
                    best_metrics[key] = min(values)
                else:
                    best_metrics[key] = max(values)
        
        return best_metrics
