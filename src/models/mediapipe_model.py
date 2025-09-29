"""
MediaPipe-based pose estimation model.
"""

import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
from .base_model import BasePoseModel


class MediaPipePose(BasePoseModel):
    """MediaPipe pose estimation model wrapper."""
    
    def __init__(
        self,
        num_keypoints: int = 33,  # MediaPipe has 33 keypoints
        input_size: Tuple[int, int] = (256, 256),
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe pose model.
        
        Args:
            num_keypoints: Number of keypoints (33 for MediaPipe)
            input_size: Input image size
            static_image_mode: Whether to use static image mode
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks
            enable_segmentation: Whether to enable segmentation
            smooth_segmentation: Whether to smooth segmentation
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        super().__init__(num_keypoints, input_size)
        
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe keypoint mapping
        self.keypoint_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - MediaPipe doesn't use traditional forward pass.
        This method is kept for compatibility.
        """
        # Convert tensor to numpy for MediaPipe processing
        if isinstance(x, torch.Tensor):
            x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Denormalize image
        x = (x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        x = np.clip(x, 0, 255).astype(np.uint8)
        
        # Process with MediaPipe
        results = self.pose.process(x)
        
        # Convert results to tensor format
        keypoints = torch.zeros(self.num_keypoints, 3)
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i, 0] = landmark.x * self.input_size[1]  # x
                keypoints[i, 1] = landmark.y * self.input_size[0]  # y
                keypoints[i, 2] = landmark.visibility  # visibility
        
        return {
            'keypoints': keypoints.unsqueeze(0),
            'confidence': torch.tensor([results.pose_landmarks is not None])
        }
    
    def predict_keypoints(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Predict keypoints from image.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            List of keypoint predictions
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Process with MediaPipe
        results = self.pose.process(image_rgb)
        
        predictions = []
        if results.pose_landmarks:
            keypoints = []
            confidences = []
            
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y])
                confidences.append(landmark.visibility)
            
            # Filter by confidence
            filtered_keypoints = []
            filtered_confidences = []
            
            for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                if conf >= confidence_threshold:
                    filtered_keypoints.append(kp)
                    filtered_confidences.append(conf)
                else:
                    filtered_keypoints.append([0, 0])  # Invalid keypoint
                    filtered_confidences.append(0.0)
            
            predictions.append({
                'keypoints': np.array(filtered_keypoints),
                'confidences': np.array(filtered_confidences),
                'landmarks': results.pose_landmarks,
                'world_landmarks': results.pose_world_landmarks,
                'segmentation_mask': results.segmentation_mask if self.enable_segmentation else None
            })
        
        return predictions
    
    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process video file for pose estimation.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of predictions for each frame
        """
        cap = cv2.VideoCapture(video_path)
        predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_predictions = self.predict_keypoints(frame)
            predictions.extend(frame_predictions)
        
        cap.release()
        return predictions
    
    def get_skeleton_connections(self) -> List[Tuple[int, int]]:
        """Get MediaPipe skeleton connections."""
        return [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            
            # Torso
            (11, 12), (11, 13), (12, 14), (13, 15),
            (11, 23), (12, 24), (23, 24),
            
            # Left arm
            (13, 15), (15, 17), (15, 19), (15, 21),
            (17, 19), (19, 21), (21, 22),
            
            # Right arm
            (14, 16), (16, 18), (16, 20), (16, 22),
            (18, 20), (20, 22),
            
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31),
            (29, 31), (31, 32),
            
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32),
            (30, 32)
        ]
    
    def visualize_results(
        self, 
        image: np.ndarray, 
        predictions: List[Dict[str, Any]],
        draw_connections: bool = True,
        draw_keypoints: bool = True
    ) -> np.ndarray:
        """
        Visualize pose estimation results.
        
        Args:
            image: Input image
            predictions: List of predictions
            draw_connections: Whether to draw skeleton connections
            draw_keypoints: Whether to draw keypoints
        
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        for prediction in predictions:
            keypoints = prediction['keypoints']
            confidences = prediction['confidences']
            
            if draw_connections:
                connections = self.get_skeleton_connections()
                for start_idx, end_idx in connections:
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                        confidences[start_idx] > 0.5 and confidences[end_idx] > 0.5):
                        
                        start_point = (
                            int(keypoints[start_idx][0] * w),
                            int(keypoints[start_idx][1] * h)
                        )
                        end_point = (
                            int(keypoints[end_idx][0] * w),
                            int(keypoints[end_idx][1] * h)
                        )
                        
                        cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
            
            if draw_keypoints:
                for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                    if conf > 0.5:
                        point = (int(kp[0] * w), int(kp[1] * h))
                        cv2.circle(vis_image, point, 5, (0, 0, 255), -1)
                        cv2.putText(
                            vis_image, 
                            str(i), 
                            (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.3, 
                            (255, 255, 255), 
                            1
                        )
        
        return vis_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get MediaPipe model information."""
        info = super().get_model_info()
        info.update({
            'model_type': 'MediaPipe',
            'static_image_mode': self.static_image_mode,
            'model_complexity': self.model_complexity,
            'smooth_landmarks': self.smooth_landmarks,
            'enable_segmentation': self.enable_segmentation,
            'min_detection_confidence': self.min_detection_confidence,
            'min_tracking_confidence': self.min_tracking_confidence
        })
        return info
