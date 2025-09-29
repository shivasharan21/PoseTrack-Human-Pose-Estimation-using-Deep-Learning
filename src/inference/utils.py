"""
Utility functions for pose detection and visualization.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[Dict[str, Any]],
    confidence_threshold: float = 0.5,
    point_size: int = 5,
    point_color: Tuple[int, int, int] = (0, 0, 255),
    show_indices: bool = False
) -> np.ndarray:
    """
    Draw keypoints on an image.
    
    Args:
        image: Input image
        keypoints: List of keypoint dictionaries
        confidence_threshold: Minimum confidence threshold
        point_size: Size of keypoint circles
        point_color: Color of keypoints (BGR)
        show_indices: Whether to show keypoint indices
    
    Returns:
        Image with keypoints drawn
    """
    vis_image = image.copy()
    
    for i, kp in enumerate(keypoints):
        if kp.get('confidence', 0) > confidence_threshold:
            x, y = int(kp['x']), int(kp['y'])
            
            # Draw keypoint
            cv2.circle(vis_image, (x, y), point_size, point_color, -1)
            
            # Draw index if requested
            if show_indices:
                cv2.putText(
                    vis_image, 
                    str(i), 
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1
                )
    
    return vis_image


def draw_skeleton(
    image: np.ndarray,
    keypoints: List[Dict[str, Any]],
    skeleton_connections: List[List[int]],
    confidence_threshold: float = 0.5,
    line_thickness: int = 2,
    line_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw skeleton connections on an image.
    
    Args:
        image: Input image
        keypoints: List of keypoint dictionaries
        skeleton_connections: List of skeleton connections
        confidence_threshold: Minimum confidence threshold
        line_thickness: Thickness of skeleton lines
        line_color: Color of skeleton lines (BGR)
    
    Returns:
        Image with skeleton drawn
    """
    vis_image = image.copy()
    
    for connection in skeleton_connections:
        if len(connection) == 2:
            start_idx, end_idx = connection[0], connection[1]
            
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                if (start_kp.get('confidence', 0) > confidence_threshold and 
                    end_kp.get('confidence', 0) > confidence_threshold):
                    
                    start_point = (int(start_kp['x']), int(start_kp['y']))
                    end_point = (int(end_kp['x']), int(end_kp['y']))
                    
                    cv2.line(vis_image, start_point, end_point, line_color, line_thickness)
    
    return vis_image


def calculate_angles(
    keypoints: List[Dict[str, Any]],
    angle_configs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate angles between keypoints.
    
    Args:
        keypoints: List of keypoint dictionaries
        angle_configs: List of angle configurations
    
    Returns:
        Dictionary of calculated angles
    """
    angles = {}
    
    for config in angle_configs:
        name = config['name']
        point1_idx = config['point1']
        vertex_idx = config['vertex']
        point2_idx = config['point2']
        
        # Get keypoints
        if (point1_idx < len(keypoints) and 
            vertex_idx < len(keypoints) and 
            point2_idx < len(keypoints)):
            
            p1 = keypoints[point1_idx]
            vertex = keypoints[vertex_idx]
            p2 = keypoints[point2_idx]
            
            # Check confidence
            if (p1.get('confidence', 0) > 0.5 and 
                vertex.get('confidence', 0) > 0.5 and 
                p2.get('confidence', 0) > 0.5):
                
                # Calculate vectors
                v1 = np.array([p1['x'] - vertex['x'], p1['y'] - vertex['y']])
                v2 = np.array([p2['x'] - vertex['x'], p2['y'] - vertex['y']])
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
                angle = np.arccos(cos_angle) * 180.0 / np.pi
                
                angles[name] = angle
            else:
                angles[name] = None  # Invalid angle
    
    return angles


def calculate_distances(
    keypoints: List[Dict[str, Any]],
    distance_configs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate distances between keypoints.
    
    Args:
        keypoints: List of keypoint dictionaries
        distance_configs: List of distance configurations
    
    Returns:
        Dictionary of calculated distances
    """
    distances = {}
    
    for config in distance_configs:
        name = config['name']
        point1_idx = config['point1']
        point2_idx = config['point2']
        
        # Get keypoints
        if (point1_idx < len(keypoints) and 
            point2_idx < len(keypoints)):
            
            p1 = keypoints[point1_idx]
            p2 = keypoints[point2_idx]
            
            # Check confidence
            if (p1.get('confidence', 0) > 0.5 and 
                p2.get('confidence', 0) > 0.5):
                
                # Calculate distance
                dx = p1['x'] - p2['x']
                dy = p1['y'] - p2['y']
                distance = np.sqrt(dx**2 + dy**2)
                
                distances[name] = distance
            else:
                distances[name] = None  # Invalid distance
    
    return distances


def normalize_pose(
    keypoints: List[Dict[str, Any]],
    reference_points: List[int] = [5, 6]  # Left and right shoulders
) -> List[Dict[str, Any]]:
    """
    Normalize pose by scaling and centering.
    
    Args:
        keypoints: List of keypoint dictionaries
        reference_points: Indices of reference points for scaling
    
    Returns:
        Normalized keypoints
    """
    if len(keypoints) == 0:
        return keypoints
    
    # Find reference points
    ref_points = []
    for ref_idx in reference_points:
        if (ref_idx < len(keypoints) and 
            keypoints[ref_idx].get('confidence', 0) > 0.5):
            ref_points.append(keypoints[ref_idx])
    
    if len(ref_points) < 2:
        return keypoints  # Cannot normalize without reference points
    
    # Calculate scale factor (use distance between reference points)
    ref_distance = np.sqrt(
        (ref_points[0]['x'] - ref_points[1]['x'])**2 + 
        (ref_points[0]['y'] - ref_points[1]['y'])**2
    )
    
    if ref_distance == 0:
        return keypoints  # Cannot normalize with zero distance
    
    # Calculate center point
    center_x = np.mean([kp['x'] for kp in ref_points])
    center_y = np.mean([kp['y'] for kp in ref_points])
    
    # Normalize keypoints
    normalized_keypoints = []
    for kp in keypoints:
        normalized_kp = kp.copy()
        
        # Center and scale
        normalized_kp['x'] = (kp['x'] - center_x) / ref_distance
        normalized_kp['y'] = (kp['y'] - center_y) / ref_distance
        
        normalized_keypoints.append(normalized_kp)
    
    return normalized_keypoints


def calculate_pose_center(keypoints: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Calculate the center of a pose.
    
    Args:
        keypoints: List of keypoint dictionaries
    
    Returns:
        Center coordinates (x, y)
    """
    valid_keypoints = [kp for kp in keypoints if kp.get('confidence', 0) > 0.5]
    
    if not valid_keypoints:
        return 0.0, 0.0
    
    center_x = np.mean([kp['x'] for kp in valid_keypoints])
    center_y = np.mean([kp['y'] for kp in valid_keypoints])
    
    return center_x, center_y


def calculate_pose_bounds(keypoints: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box of a pose.
    
    Args:
        keypoints: List of keypoint dictionaries
    
    Returns:
        Bounding box (x_min, y_min, x_max, y_max)
    """
    valid_keypoints = [kp for kp in keypoints if kp.get('confidence', 0) > 0.5]
    
    if not valid_keypoints:
        return 0.0, 0.0, 0.0, 0.0
    
    x_coords = [kp['x'] for kp in valid_keypoints]
    y_coords = [kp['y'] for kp in valid_keypoints]
    
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def filter_keypoints_by_confidence(
    keypoints: List[Dict[str, Any]],
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter keypoints by confidence threshold.
    
    Args:
        keypoints: List of keypoint dictionaries
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        Filtered keypoints
    """
    return [kp for kp in keypoints if kp.get('confidence', 0) > confidence_threshold]


def interpolate_missing_keypoints(
    keypoints: List[Dict[str, Any]],
    max_gap: int = 3
) -> List[Dict[str, Any]]:
    """
    Interpolate missing keypoints using temporal information.
    
    Args:
        keypoints: List of keypoint dictionaries
        max_gap: Maximum gap size to interpolate
    
    Returns:
        Keypoints with interpolated values
    """
    if len(keypoints) < 2:
        return keypoints
    
    interpolated = keypoints.copy()
    
    for i in range(len(keypoints)):
        kp = keypoints[i]
        
        if kp.get('confidence', 0) < 0.5:
            # Find nearest valid keypoints
            prev_valid = None
            next_valid = None
            
            # Look backward
            for j in range(i - 1, max(0, i - max_gap) - 1, -1):
                if keypoints[j].get('confidence', 0) > 0.5:
                    prev_valid = j
                    break
            
            # Look forward
            for j in range(i + 1, min(len(keypoints), i + max_gap + 1)):
                if keypoints[j].get('confidence', 0) > 0.5:
                    next_valid = j
                    break
            
            # Interpolate if we have both neighbors
            if prev_valid is not None and next_valid is not None:
                prev_kp = keypoints[prev_valid]
                next_kp = keypoints[next_valid]
                
                # Linear interpolation
                t = (i - prev_valid) / (next_valid - prev_valid)
                
                interpolated[i]['x'] = prev_kp['x'] + t * (next_kp['x'] - prev_kp['x'])
                interpolated[i]['y'] = prev_kp['y'] + t * (next_kp['y'] - prev_kp['y'])
                interpolated[i]['confidence'] = prev_kp['confidence'] + t * (next_kp['confidence'] - prev_kp['confidence'])
    
    return interpolated


def calculate_pose_velocity(
    keypoints_sequence: List[List[Dict[str, Any]]],
    fps: float = 30.0
) -> List[Dict[str, float]]:
    """
    Calculate velocity of keypoints over time.
    
    Args:
        keypoints_sequence: Sequence of keypoint lists
        fps: Frames per second
    
    Returns:
        List of velocity dictionaries
    """
    if len(keypoints_sequence) < 2:
        return []
    
    velocities = []
    
    for i in range(1, len(keypoints_sequence)):
        prev_keypoints = keypoints_sequence[i - 1]
        curr_keypoints = keypoints_sequence[i]
        
        frame_velocities = {}
        
        for j in range(min(len(prev_keypoints), len(curr_keypoints))):
            prev_kp = prev_keypoints[j]
            curr_kp = curr_keypoints[j]
            
            if (prev_kp.get('confidence', 0) > 0.5 and 
                curr_kp.get('confidence', 0) > 0.5):
                
                # Calculate velocity (pixels per second)
                dx = curr_kp['x'] - prev_kp['x']
                dy = curr_kp['y'] - prev_kp['y']
                
                velocity_x = dx * fps
                velocity_y = dy * fps
                velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                
                frame_velocities[f'kp_{j}'] = {
                    'velocity_x': velocity_x,
                    'velocity_y': velocity_y,
                    'velocity_magnitude': velocity_magnitude
                }
        
        velocities.append(frame_velocities)
    
    return velocities


def smooth_keypoints_trajectory(
    keypoints_sequence: List[List[Dict[str, Any]]],
    window_size: int = 5
) -> List[List[Dict[str, Any]]]:
    """
    Apply temporal smoothing to keypoint trajectories.
    
    Args:
        keypoints_sequence: Sequence of keypoint lists
        window_size: Size of smoothing window
    
    Returns:
        Smoothed keypoint sequence
    """
    if len(keypoints_sequence) < window_size:
        return keypoints_sequence
    
    smoothed_sequence = []
    
    for i in range(len(keypoints_sequence)):
        smoothed_keypoints = []
        
        # Get window of keypoints around current frame
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(keypoints_sequence), i + window_size // 2 + 1)
        
        window_keypoints = keypoints_sequence[start_idx:end_idx]
        
        # Smooth each keypoint
        for j in range(len(keypoints_sequence[i])):
            valid_positions = []
            
            for frame_keypoints in window_keypoints:
                if (j < len(frame_keypoints) and 
                    frame_keypoints[j].get('confidence', 0) > 0.5):
                    valid_positions.append([
                        frame_keypoints[j]['x'],
                        frame_keypoints[j]['y']
                    ])
            
            if valid_positions:
                # Calculate smoothed position
                smoothed_x = np.mean([pos[0] for pos in valid_positions])
                smoothed_y = np.mean([pos[1] for pos in valid_positions])
                
                # Create smoothed keypoint
                smoothed_kp = keypoints_sequence[i][j].copy()
                smoothed_kp['x'] = smoothed_x
                smoothed_kp['y'] = smoothed_y
                smoothed_keypoints.append(smoothed_kp)
            else:
                smoothed_keypoints.append(keypoints_sequence[i][j])
        
        smoothed_sequence.append(smoothed_keypoints)
    
    return smoothed_sequence
