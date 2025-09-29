"""
Utility functions for pose estimation models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from scipy.ndimage import gaussian_filter


def create_heatmap(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
    sigma: float = 2.0,
    heatmap_type: str = 'gaussian'
) -> np.ndarray:
    """
    Create heatmap from keypoints.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        image_size: Target heatmap size (height, width)
        sigma: Gaussian sigma for heatmap generation
        heatmap_type: Type of heatmap ('gaussian', 'uniform')
    
    Returns:
        Heatmap array [N, height, width]
    """
    heatmaps = []
    
    for i, (x, y, visibility) in enumerate(keypoints):
        if visibility > 0.5:
            if heatmap_type == 'gaussian':
                heatmap = _create_gaussian_heatmap(x, y, image_size, sigma)
            elif heatmap_type == 'uniform':
                heatmap = _create_uniform_heatmap(x, y, image_size)
            else:
                raise ValueError(f"Unknown heatmap type: {heatmap_type}")
            
            heatmap *= visibility
        else:
            heatmap = np.zeros(image_size)
        
        heatmaps.append(heatmap)
    
    return np.array(heatmaps)


def _create_gaussian_heatmap(
    center_x: float, 
    center_y: float, 
    image_size: Tuple[int, int], 
    sigma: float
) -> np.ndarray:
    """Create Gaussian heatmap centered at (center_x, center_y)."""
    height, width = image_size
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Calculate Gaussian
    heatmap = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
    
    return heatmap


def _create_uniform_heatmap(
    center_x: float, 
    center_y: float, 
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Create uniform heatmap centered at (center_x, center_y)."""
    height, width = image_size
    heatmap = np.zeros((height, width))
    
    # Set a small region around the center to 1
    radius = 3
    y_min = max(0, int(center_y) - radius)
    y_max = min(height, int(center_y) + radius + 1)
    x_min = max(0, int(center_x) - radius)
    x_max = min(width, int(center_x) + radius + 1)
    
    heatmap[y_min:y_max, x_min:x_max] = 1.0
    
    return heatmap


def decode_heatmap(
    heatmap: np.ndarray,
    confidence_threshold: float = 0.5,
    use_nms: bool = True,
    nms_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Decode heatmap to keypoints.
    
    Args:
        heatmap: Heatmap array [C, H, W]
        confidence_threshold: Minimum confidence threshold
        use_nms: Whether to use non-maximum suppression
        nms_threshold: NMS threshold
    
    Returns:
        List of keypoint predictions
    """
    predictions = []
    
    for i in range(heatmap.shape[0]):
        hm = heatmap[i]
        
        if use_nms:
            peaks = _find_peaks_with_nms(hm, confidence_threshold, nms_threshold)
        else:
            peaks = _find_peaks_simple(hm, confidence_threshold)
        
        for peak in peaks:
            y, x, confidence = peak
            predictions.append({
                'keypoint_id': i,
                'x': x,
                'y': y,
                'confidence': confidence
            })
    
    return predictions


def _find_peaks_simple(heatmap: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """Find peaks in heatmap using simple approach."""
    from scipy.ndimage import maximum_filter
    
    # Apply non-maximum suppression
    local_maxima = maximum_filter(heatmap, size=3) == heatmap
    
    peaks = []
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            if local_maxima[y, x] and heatmap[y, x] > threshold:
                peaks.append((y, x, heatmap[y, x]))
    
    return peaks


def _find_peaks_with_nms(
    heatmap: np.ndarray, 
    confidence_threshold: float, 
    nms_threshold: float
) -> List[Tuple[int, int, float]]:
    """Find peaks in heatmap using NMS."""
    from scipy.ndimage import maximum_filter
    
    # Apply non-maximum suppression
    local_maxima = maximum_filter(heatmap, size=3) == heatmap
    
    # Get candidate peaks
    candidates = []
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            if local_maxima[y, x] and heatmap[y, x] > confidence_threshold:
                candidates.append((heatmap[y, x], y, x))
    
    # Sort by confidence
    candidates.sort(reverse=True)
    
    # Apply NMS
    peaks = []
    for confidence, y, x in candidates:
        # Check if this peak is too close to existing peaks
        too_close = False
        for existing_confidence, existing_y, existing_x in peaks:
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < nms_threshold:
                too_close = True
                break
        
        if not too_close:
            peaks.append((y, x, confidence))
    
    return peaks


def create_paf_vector(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    image_size: Tuple[int, int],
    sigma: float = 1.0
) -> np.ndarray:
    """
    Create Part Affinity Field vector between two points.
    
    Args:
        start_point: Start point (x, y)
        end_point: End point (x, y)
        image_size: Image size (height, width)
        sigma: Gaussian sigma
    
    Returns:
        PAF vector field [2, height, width]
    """
    height, width = image_size
    paf = np.zeros((2, height, width))
    
    # Calculate vector from start to end
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return paf
    
    # Normalize vector
    dx /= length
    dy /= length
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    # Calculate distance from line segment
    t = ((x_coords - start_point[0]) * dx + (y_coords - start_point[1]) * dy) / length
    t = np.clip(t, 0, 1)
    
    # Closest point on line segment
    closest_x = start_point[0] + t * dx * length
    closest_y = start_point[1] + t * dy * length
    
    # Distance from line segment
    dist = np.sqrt((x_coords - closest_x)**2 + (y_coords - closest_y)**2)
    
    # Create Gaussian mask
    mask = np.exp(-(dist**2) / (2 * sigma**2))
    
    # Set PAF vectors
    paf[0] = dx * mask
    paf[1] = dy * mask
    
    return paf


def decode_paf(
    paf: np.ndarray,
    confidence_map: np.ndarray,
    skeleton_connections: List[Tuple[int, int]],
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Decode Part Affinity Field to skeleton connections.
    
    Args:
        paf: PAF array [C, H, W]
        confidence_map: Confidence map array [C, H, W]
        skeleton_connections: List of skeleton connections
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        List of skeleton connections
    """
    connections = []
    
    for connection_idx, (start_id, end_id) in enumerate(skeleton_connections):
        # Get PAF channels for this connection
        paf_x = paf[connection_idx * 2]
        paf_y = paf[connection_idx * 2 + 1]
        
        # Get confidence maps for start and end points
        start_cm = confidence_map[start_id]
        end_cm = confidence_map[end_id]
        
        # Find peaks in confidence maps
        start_peaks = _find_peaks_simple(start_cm, confidence_threshold)
        end_peaks = _find_peaks_simple(end_cm, confidence_threshold)
        
        # Find best connections
        for start_peak in start_peaks:
            for end_peak in end_peaks:
                start_y, start_x, start_conf = start_peak
                end_y, end_x, end_conf = end_peak
                
                # Calculate PAF score
                paf_score = _calculate_paf_score(
                    (start_x, start_y), (end_x, end_y), paf_x, paf_y
                )
                
                if paf_score > 0.1:  # Minimum PAF score
                    connections.append({
                        'start_keypoint_id': start_id,
                        'end_keypoint_id': end_id,
                        'start_point': (start_x, start_y),
                        'end_point': (end_x, end_y),
                        'start_confidence': start_conf,
                        'end_confidence': end_conf,
                        'paf_score': paf_score
                    })
    
    return connections


def _calculate_paf_score(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    paf_x: np.ndarray,
    paf_y: np.ndarray
) -> float:
    """Calculate PAF score between two points."""
    # Sample points along the line
    num_samples = 10
    scores = []
    
    for i in range(num_samples):
        t = i / (num_samples - 1)
        x = int(start_point[0] * (1 - t) + end_point[0] * t)
        y = int(start_point[1] * (1 - t) + end_point[1] * t)
        
        # Check bounds
        if 0 <= y < paf_x.shape[0] and 0 <= x < paf_x.shape[1]:
            # Calculate expected vector
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx /= length
                dy /= length
                
                # Get PAF values
                paf_val_x = paf_x[y, x]
                paf_val_y = paf_y[y, x]
                
                # Calculate dot product
                score = dx * paf_val_x + dy * paf_val_y
                scores.append(score)
    
    return np.mean(scores) if scores else 0


def visualize_heatmap(
    heatmap: np.ndarray,
    image: Optional[np.ndarray] = None,
    alpha: float = 0.6,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Visualize heatmap overlay on image.
    
    Args:
        heatmap: Heatmap array [H, W] or [C, H, W]
        image: Original image (optional)
        alpha: Overlay transparency
        colormap: Colormap for heatmap
    
    Returns:
        Visualization image
    """
    if heatmap.ndim == 3:
        # Take maximum across channels
        heatmap = np.max(heatmap, axis=0)
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply colormap
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    if image is not None:
        # Resize heatmap to match image
        if heatmap_colored.shape[:2] != image.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        # Blend with original image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    else:
        result = heatmap_colored
    
    return result


def calculate_pose_metrics(
    predicted_keypoints: List[Dict[str, Any]],
    ground_truth_keypoints: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate pose estimation metrics.
    
    Args:
        predicted_keypoints: List of predicted keypoints
        ground_truth_keypoints: List of ground truth keypoints
        threshold: PCK threshold
    
    Returns:
        Dictionary of metrics
    """
    if not predicted_keypoints or not ground_truth_keypoints:
        return {'pck': 0.0, 'mpjpe': float('inf')}
    
    # Calculate PCK (Percentage of Correct Keypoints)
    correct_keypoints = 0
    total_keypoints = 0
    
    # Calculate MPJPE (Mean Per Joint Position Error)
    position_errors = []
    
    for pred_kp, gt_kp in zip(predicted_keypoints, ground_truth_keypoints):
        if gt_kp['confidence'] > 0.5:  # Only consider visible GT keypoints
            pred_x, pred_y = pred_kp['x'], pred_kp['y']
            gt_x, gt_y = gt_kp['x'], gt_kp['y']
            
            # Calculate distance
            distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            position_errors.append(distance)
            
            # Check if within threshold
            if distance < threshold:
                correct_keypoints += 1
            
            total_keypoints += 1
    
    pck = correct_keypoints / total_keypoints if total_keypoints > 0 else 0.0
    mpjpe = np.mean(position_errors) if position_errors else float('inf')
    
    return {
        'pck': pck,
        'mpjpe': mpjpe,
        'total_keypoints': total_keypoints,
        'correct_keypoints': correct_keypoints
    }
