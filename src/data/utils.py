"""
Utility functions for pose estimation data processing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict, Optional, Any
import os
from PIL import Image


# COCO keypoint skeleton connections
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# COCO keypoint names
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# MPII keypoint skeleton connections
MPII_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6],
    [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
    [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]
]

# MPII keypoint names
MPII_KEYPOINT_NAMES = [
    'right_ankle', 'right_knee', 'right_hip', 'left_hip',
    'left_knee', 'left_ankle', 'pelvis', 'thorax', 'upper_neck',
    'head_top', 'right_wrist', 'right_elbow', 'right_shoulder',
    'left_shoulder', 'left_elbow', 'left_wrist', 'left_ankle',
    'left_knee', 'left_hip', 'right_hip', 'right_knee'
]


def visualize_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: Optional[List[List[int]]] = None,
    keypoint_names: Optional[List[str]] = None,
    visibility_threshold: float = 0.5,
    point_size: int = 5,
    line_thickness: int = 2,
    show_names: bool = False,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize keypoints on an image.
    
    Args:
        image: Input image
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        skeleton: List of skeleton connections
        keypoint_names: List of keypoint names
        visibility_threshold: Minimum visibility for drawing
        point_size: Size of keypoint circles
        line_thickness: Thickness of skeleton lines
        show_names: Whether to show keypoint names
        save_path: Path to save the visualization
    
    Returns:
        Image with keypoints visualized
    """
    if skeleton is None:
        skeleton = COCO_SKELETON
    
    if keypoint_names is None:
        keypoint_names = COCO_KEYPOINT_NAMES
    
    # Create a copy of the image
    vis_image = image.copy()
    
    # Draw skeleton connections
    for connection in skeleton:
        if len(connection) == 2:
            start_idx, end_idx = connection[0], connection[1]
            
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx, 2] > visibility_threshold and
                keypoints[end_idx, 2] > visibility_threshold):
                
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                
                cv2.line(vis_image, start_point, end_point, (0, 255, 0), line_thickness)
    
    # Draw keypoints
    for i, (x, y, visibility) in enumerate(keypoints):
        if visibility > visibility_threshold:
            point = (int(x), int(y))
            cv2.circle(vis_image, point, point_size, (0, 0, 255), -1)
            
            if show_names and i < len(keypoint_names):
                cv2.putText(
                    vis_image, 
                    keypoint_names[i], 
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1
                )
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image


def load_annotations(annotations_path: str) -> Dict[str, Any]:
    """
    Load annotations from JSON file.
    
    Args:
        annotations_path: Path to annotations JSON file
    
    Returns:
        Dictionary containing annotations
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def save_annotations(annotations: Dict[str, Any], save_path: str) -> None:
    """
    Save annotations to JSON file.
    
    Args:
        annotations: Annotations dictionary
        save_path: Path to save annotations
    """
    with open(save_path, 'w') as f:
        json.dump(annotations, f, indent=2)


def convert_keypoints_format(
    keypoints: np.ndarray,
    from_format: str,
    to_format: str
) -> np.ndarray:
    """
    Convert keypoints between different formats.
    
    Args:
        keypoints: Input keypoints array
        from_format: Source format ('coco', 'mpii')
        to_format: Target format ('coco', 'mpii')
    
    Returns:
        Converted keypoints array
    """
    if from_format == to_format:
        return keypoints
    
    # Add conversion logic here based on specific requirements
    # This is a placeholder for format conversion
    
    return keypoints


def calculate_keypoint_distances(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate distances between keypoints.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
    
    Returns:
        Dictionary of keypoint distances
    """
    distances = {}
    
    # Calculate some common distances
    if len(keypoints) >= 17:  # COCO format
        # Shoulder width
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
            distances['shoulder_width'] = np.linalg.norm(
                left_shoulder[:2] - right_shoulder[:2]
            )
        
        # Hip width
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        if left_hip[2] > 0.5 and right_hip[2] > 0.5:
            distances['hip_width'] = np.linalg.norm(
                left_hip[:2] - right_hip[:2]
            )
        
        # Arm length (left)
        left_shoulder = keypoints[5]
        left_wrist = keypoints[9]
        if left_shoulder[2] > 0.5 and left_wrist[2] > 0.5:
            distances['left_arm_length'] = np.linalg.norm(
                left_shoulder[:2] - left_wrist[:2]
            )
        
        # Leg length (left)
        left_hip = keypoints[11]
        left_ankle = keypoints[15]
        if left_hip[2] > 0.5 and left_ankle[2] > 0.5:
            distances['left_leg_length'] = np.linalg.norm(
                left_hip[:2] - left_ankle[:2]
            )
    
    return distances


def filter_keypoints_by_visibility(
    keypoints: np.ndarray,
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """
    Filter keypoints by visibility threshold.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        visibility_threshold: Minimum visibility threshold
    
    Returns:
        Filtered keypoints array
    """
    visible_mask = keypoints[:, 2] > visibility_threshold
    return keypoints[visible_mask]


def normalize_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize keypoints to [0, 1] range.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        image_size: Image size (height, width)
    
    Returns:
        Normalized keypoints array
    """
    normalized = keypoints.copy()
    normalized[:, 0] /= image_size[1]  # x / width
    normalized[:, 1] /= image_size[0]  # y / height
    return normalized


def denormalize_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Denormalize keypoints from [0, 1] range to pixel coordinates.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        image_size: Image size (height, width)
    
    Returns:
        Denormalized keypoints array
    """
    denormalized = keypoints.copy()
    denormalized[:, 0] *= image_size[1]  # x * width
    denormalized[:, 1] *= image_size[0]  # y * height
    return denormalized


def create_heatmap(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
    sigma: float = 2.0
) -> np.ndarray:
    """
    Create heatmap from keypoints.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        image_size: Target heatmap size (height, width)
        sigma: Gaussian sigma for heatmap generation
    
    Returns:
        Heatmap array [N, height, width]
    """
    heatmaps = []
    
    for i, (x, y, visibility) in enumerate(keypoints):
        if visibility > 0.5:
            heatmap = np.zeros(image_size)
            
            # Create 2D Gaussian
            y_coords, x_coords = np.ogrid[:image_size[0], :image_size[1]]
            heatmap = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
            heatmap *= visibility
        else:
            heatmap = np.zeros(image_size)
        
        heatmaps.append(heatmap)
    
    return np.array(heatmaps)


def plot_keypoints_statistics(
    annotations: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Plot statistics about keypoints in the dataset.
    
    Args:
        annotations: Annotations dictionary
        save_path: Path to save the plot
    """
    # Extract keypoint statistics
    keypoint_counts = []
    visibility_scores = []
    
    for annotation in annotations['annotations']:
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
        visible_keypoints = np.sum(keypoints[:, 2] > 0.5)
        avg_visibility = np.mean(keypoints[:, 2])
        
        keypoint_counts.append(visible_keypoints)
        visibility_scores.append(avg_visibility)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Keypoint count distribution
    ax1.hist(keypoint_counts, bins=20, alpha=0.7, color='blue')
    ax1.set_xlabel('Number of Visible Keypoints')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Visible Keypoints')
    ax1.grid(True, alpha=0.3)
    
    # Visibility score distribution
    ax2.hist(visibility_scores, bins=20, alpha=0.7, color='green')
    ax2.set_xlabel('Average Visibility Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Visibility Scores')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def validate_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
    tolerance: float = 0.1
) -> bool:
    """
    Validate keypoints for correctness.
    
    Args:
        keypoints: Array of keypoints [N, 3] (x, y, visibility)
        image_size: Image size (height, width)
        tolerance: Tolerance for boundary checks
    
    Returns:
        True if keypoints are valid, False otherwise
    """
    # Check if keypoints are within image bounds
    valid_x = np.all((keypoints[:, 0] >= -tolerance) & 
                     (keypoints[:, 0] <= image_size[1] + tolerance))
    valid_y = np.all((keypoints[:, 1] >= -tolerance) & 
                     (keypoints[:, 1] <= image_size[0] + tolerance))
    valid_visibility = np.all((keypoints[:, 2] >= 0) & (keypoints[:, 2] <= 1))
    
    return valid_x and valid_y and valid_visibility
