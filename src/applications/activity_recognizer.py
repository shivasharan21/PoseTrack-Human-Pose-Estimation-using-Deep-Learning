"""
Activity recognition and gesture detection application.
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import deque
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

from ..inference import PoseDetector
from ..inference.utils import calculate_angles, calculate_distances, normalize_pose


@dataclass
class Gesture:
    """Data class for a detected gesture."""
    gesture_type: str
    confidence: float
    start_time: float
    end_time: float
    keypoints_sequence: List[List[Dict[str, Any]]]
    features: List[float]


@dataclass
class Activity:
    """Data class for a detected activity."""
    activity_type: str
    confidence: float
    start_time: float
    end_time: float
    gestures: List[Gesture]
    features: List[float]


class FeatureExtractor:
    """Extracts features from pose keypoints for activity recognition."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            'head_velocity', 'shoulder_velocity', 'hip_velocity',
            'elbow_angles', 'knee_angles', 'wrist_angles',
            'body_center_x', 'body_center_y', 'body_width', 'body_height',
            'arm_span', 'leg_span', 'torso_length'
        ]
    
    def extract_features(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """
        Extract features from keypoint sequence.
        
        Args:
            keypoints_sequence: Sequence of keypoint lists
        
        Returns:
            List of extracted features
        """
        if not keypoints_sequence:
            return [0.0] * len(self.feature_names)
        
        features = []
        
        # Velocity features
        velocity_features = self._extract_velocity_features(keypoints_sequence)
        features.extend(velocity_features)
        
        # Angle features
        angle_features = self._extract_angle_features(keypoints_sequence)
        features.extend(angle_features)
        
        # Body dimension features
        dimension_features = self._extract_dimension_features(keypoints_sequence)
        features.extend(dimension_features)
        
        return features
    
    def _extract_velocity_features(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """Extract velocity-based features."""
        features = []
        
        if len(keypoints_sequence) < 2:
            return [0.0, 0.0, 0.0]  # head, shoulder, hip velocity
        
        # Calculate velocities for key body parts
        keypoints_to_track = [0, 5, 11]  # nose, left_shoulder, left_hip
        
        for kp_idx in keypoints_to_track:
            velocities = []
            
            for i in range(1, len(keypoints_sequence)):
                prev_kp = keypoints_sequence[i-1]
                curr_kp = keypoints_sequence[i]
                
                if (kp_idx < len(prev_kp) and kp_idx < len(curr_kp) and
                    prev_kp[kp_idx].get('confidence', 0) > 0.5 and
                    curr_kp[kp_idx].get('confidence', 0) > 0.5):
                    
                    dx = curr_kp[kp_idx]['x'] - prev_kp[kp_idx]['x']
                    dy = curr_kp[kp_idx]['y'] - prev_kp[kp_idx]['y']
                    velocity = np.sqrt(dx**2 + dy**2)
                    velocities.append(velocity)
            
            avg_velocity = np.mean(velocities) if velocities else 0.0
            features.append(avg_velocity)
        
        return features
    
    def _extract_angle_features(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """Extract angle-based features."""
        features = []
        
        if not keypoints_sequence:
            return [0.0, 0.0, 0.0]  # elbow, knee, wrist angles
        
        # Calculate average angles for key joints
        angle_configs = [
            {'name': 'elbow_angle', 'point1': 5, 'vertex': 7, 'point2': 9},  # left elbow
            {'name': 'knee_angle', 'point1': 11, 'vertex': 13, 'point2': 15},  # left knee
            {'name': 'wrist_angle', 'point1': 7, 'vertex': 9, 'point2': 10},  # wrist angle
        ]
        
        for config in angle_configs:
            angles = []
            
            for keypoints in keypoints_sequence:
                angle_result = calculate_angles([keypoints], [config])
                if config['name'] in angle_result[0]:
                    angles.append(angle_result[0][config['name']])
            
            avg_angle = np.mean([a for a in angles if a is not None]) if angles else 0.0
            features.append(avg_angle)
        
        return features
    
    def _extract_dimension_features(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> List[float]:
        """Extract body dimension features."""
        features = []
        
        if not keypoints_sequence:
            return [0.0] * 7  # 7 dimension features
        
        # Use the first frame for dimension calculations
        keypoints = keypoints_sequence[0]
        
        # Body center
        valid_keypoints = [kp for kp in keypoints if kp.get('confidence', 0) > 0.5]
        if valid_keypoints:
            center_x = np.mean([kp['x'] for kp in valid_keypoints])
            center_y = np.mean([kp['y'] for kp in valid_keypoints])
        else:
            center_x = center_y = 0.0
        
        features.extend([center_x, center_y])
        
        # Body dimensions
        body_width = self._calculate_body_width(keypoints)
        body_height = self._calculate_body_height(keypoints)
        arm_span = self._calculate_arm_span(keypoints)
        leg_span = self._calculate_leg_span(keypoints)
        torso_length = self._calculate_torso_length(keypoints)
        
        features.extend([body_width, body_height, arm_span, leg_span, torso_length])
        
        return features
    
    def _calculate_body_width(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate body width."""
        left_shoulder = self._get_keypoint(keypoints, 5)
        right_shoulder = self._get_keypoint(keypoints, 6)
        
        if left_shoulder and right_shoulder:
            return abs(right_shoulder['x'] - left_shoulder['x'])
        return 0.0
    
    def _calculate_body_height(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate body height."""
        head = self._get_keypoint(keypoints, 0)  # nose
        left_ankle = self._get_keypoint(keypoints, 15)
        
        if head and left_ankle:
            return abs(left_ankle['y'] - head['y'])
        return 0.0
    
    def _calculate_arm_span(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate arm span."""
        left_wrist = self._get_keypoint(keypoints, 9)
        right_wrist = self._get_keypoint(keypoints, 10)
        
        if left_wrist and right_wrist:
            return abs(right_wrist['x'] - left_wrist['x'])
        return 0.0
    
    def _calculate_leg_span(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate leg span."""
        left_ankle = self._get_keypoint(keypoints, 15)
        right_ankle = self._get_keypoint(keypoints, 16)
        
        if left_ankle and right_ankle:
            return abs(right_ankle['x'] - left_ankle['x'])
        return 0.0
    
    def _calculate_torso_length(self, keypoints: List[Dict[str, Any]]) -> float:
        """Calculate torso length."""
        left_shoulder = self._get_keypoint(keypoints, 5)
        left_hip = self._get_keypoint(keypoints, 11)
        
        if left_shoulder and left_hip:
            return abs(left_hip['y'] - left_shoulder['y'])
        return 0.0
    
    def _get_keypoint(self, keypoints: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
        """Get keypoint by index with confidence check."""
        if index < len(keypoints):
            kp = keypoints[index]
            if kp.get('confidence', 0) > 0.5:
                return kp
        return None


class GestureDetector:
    """Detects specific gestures from pose sequences."""
    
    def __init__(self):
        """Initialize gesture detector."""
        self.gesture_templates = {
            'wave': self._create_wave_template(),
            'thumbs_up': self._create_thumbs_up_template(),
            'clap': self._create_clap_template(),
            'point': self._create_point_template(),
            'stop': self._create_stop_template()
        }
        
        self.feature_extractor = FeatureExtractor()
    
    def detect_gesture(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> Optional[Gesture]:
        """
        Detect gesture from keypoint sequence.
        
        Args:
            keypoints_sequence: Sequence of keypoints
        
        Returns:
            Detected gesture or None
        """
        if not keypoints_sequence:
            return None
        
        # Extract features
        features = self.feature_extractor.extract_features(keypoints_sequence)
        
        # Compare with gesture templates
        best_gesture = None
        best_confidence = 0.0
        
        for gesture_type, template in self.gesture_templates.items():
            confidence = self._calculate_gesture_similarity(features, template)
            
            if confidence > best_confidence and confidence > 0.7:  # Threshold
                best_confidence = confidence
                best_gesture = gesture_type
        
        if best_gesture:
            return Gesture(
                gesture_type=best_gesture,
                confidence=best_confidence,
                start_time=time.time() - len(keypoints_sequence) * 0.1,  # Estimate
                end_time=time.time(),
                keypoints_sequence=keypoints_sequence,
                features=features
            )
        
        return None
    
    def _calculate_gesture_similarity(
        self, 
        features: List[float], 
        template: Dict[str, Any]
    ) -> float:
        """Calculate similarity between features and gesture template."""
        # Simple template matching - in practice, use more sophisticated methods
        template_features = template['features']
        
        if len(features) != len(template_features):
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(features, template_features)
        norm_features = np.linalg.norm(features)
        norm_template = np.linalg.norm(template_features)
        
        if norm_features == 0 or norm_template == 0:
            return 0.0
        
        similarity = dot_product / (norm_features * norm_template)
        return similarity
    
    def _create_wave_template(self) -> Dict[str, Any]:
        """Create wave gesture template."""
        # Simplified template - in practice, use real gesture data
        features = [0.5, 0.3, 0.1, 120, 180, 90, 320, 240, 60, 180, 80, 40, 50]
        return {'features': features, 'description': 'Hand waving motion'}
    
    def _create_thumbs_up_template(self) -> Dict[str, Any]:
        """Create thumbs up gesture template."""
        features = [0.1, 0.2, 0.1, 150, 160, 120, 320, 220, 50, 170, 60, 30, 45]
        return {'features': features, 'description': 'Thumbs up gesture'}
    
    def _create_clap_template(self) -> Dict[str, Any]:
        """Create clap gesture template."""
        features = [0.2, 0.4, 0.1, 100, 170, 80, 320, 230, 45, 175, 40, 35, 48]
        return {'features': features, 'description': 'Clapping motion'}
    
    def _create_point_template(self) -> Dict[str, Any]:
        """Create pointing gesture template."""
        features = [0.1, 0.3, 0.1, 140, 175, 100, 320, 225, 55, 180, 70, 35, 47]
        return {'features': features, 'description': 'Pointing gesture'}
    
    def _create_stop_template(self) -> Dict[str, Any]:
        """Create stop gesture template."""
        features = [0.0, 0.1, 0.0, 180, 180, 180, 320, 240, 60, 180, 80, 40, 50]
        return {'features': features, 'description': 'Stop gesture'}


class ActivityRecognizer:
    """Recognizes activities from pose sequences using machine learning."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize activity recognizer.
        
        Args:
            model_path: Path to pre-trained model
        """
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Activity classes
        self.activity_classes = [
            'walking', 'running', 'sitting', 'standing', 'jumping',
            'dancing', 'exercising', 'cooking', 'reading', 'working'
        ]
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with a simple default model."""
        # Use a simple classifier for demonstration
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create dummy training data for initialization
        # In practice, this would be trained on real data
        dummy_features = np.random.randn(100, len(self.feature_extractor.feature_names))
        dummy_labels = np.random.choice(self.activity_classes, 100)
        
        dummy_features_scaled = self.scaler.fit_transform(dummy_features)
        self.classifier.fit(dummy_features_scaled, dummy_labels)
    
    def recognize_activity(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> Optional[Activity]:
        """
        Recognize activity from keypoint sequence.
        
        Args:
            keypoints_sequence: Sequence of keypoints
        
        Returns:
            Recognized activity or None
        """
        if not keypoints_sequence or not self.classifier:
            return None
        
        # Extract features
        features = self.feature_extractor.extract_features(keypoints_sequence)
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Predict activity
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = np.max(self.classifier.predict_proba(features_scaled))
        
        if confidence > 0.6:  # Confidence threshold
            return Activity(
                activity_type=prediction,
                confidence=confidence,
                start_time=time.time() - len(keypoints_sequence) * 0.1,
                end_time=time.time(),
                gestures=[],  # Would be populated by gesture detector
                features=features
            )
        
        return None
    
    def train_model(
        self, 
        training_data: List[Tuple[List[List[Dict[str, Any]]], str]]
    ):
        """
        Train the activity recognition model.
        
        Args:
            training_data: List of (keypoints_sequence, activity_label) tuples
        """
        if not training_data:
            raise ValueError("No training data provided")
        
        # Extract features and labels
        features_list = []
        labels = []
        
        for keypoints_sequence, activity_label in training_data:
            features = self.feature_extractor.extract_features(keypoints_sequence)
            features_list.append(features)
            labels.append(activity_label)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_scaled, y)
        
        print(f"Model trained on {len(training_data)} samples")
        print(f"Classes: {self.classifier.classes_}")
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'activity_classes': self.activity_classes
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.activity_classes = model_data['activity_classes']
        
        print(f"Model loaded from: {model_path}")


class RealTimeActivityTracker:
    """Real-time activity tracking application."""
    
    def __init__(self, model_type: str = 'mediapipe'):
        """
        Initialize real-time activity tracker.
        
        Args:
            model_type: Type of pose estimation model to use
        """
        self.detector = PoseDetector(model_type=model_type)
        self.activity_recognizer = ActivityRecognizer()
        self.gesture_detector = GestureDetector()
        
        # Tracking state
        self.keypoint_buffer = deque(maxlen=30)  # Buffer for recent keypoints
        self.current_activity = None
        self.current_gesture = None
        self.is_tracking = False
        
        # Activity history
        self.activity_history = deque(maxlen=100)
        self.gesture_history = deque(maxlen=50)
    
    def start_tracking(self):
        """Start activity tracking."""
        self.is_tracking = True
        self.keypoint_buffer.clear()
        print("Started activity tracking")
    
    def stop_tracking(self):
        """Stop activity tracking."""
        self.is_tracking = False
        print("Stopped activity tracking")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for activity recognition.
        
        Args:
            frame: Input video frame
        
        Returns:
            Dictionary containing recognition results
        """
        if not self.is_tracking:
            return {'status': 'not_tracking'}
        
        # Detect poses
        results = self.detector.detect_poses(frame, return_visualization=False)
        
        if not results['predictions']:
            return {'status': 'no_pose_detected'}
        
        # Get first person's keypoints
        keypoints = results['predictions'][0].get('keypoints', [])
        
        # Add to buffer
        self.keypoint_buffer.append(keypoints)
        
        # Recognize activity if we have enough frames
        activity_result = None
        if len(self.keypoint_buffer) >= 10:
            activity_result = self.activity_recognizer.recognize_activity(
                list(self.keypoint_buffer)
            )
            
            if activity_result:
                self.current_activity = activity_result
                self.activity_history.append(activity_result)
        
        # Detect gestures
        gesture_result = None
        if len(self.keypoint_buffer) >= 5:
            gesture_result = self.gesture_detector.detect_gesture(
                list(self.keypoint_buffer)
            )
            
            if gesture_result:
                self.current_gesture = gesture_result
                self.gesture_history.append(gesture_result)
        
        return {
            'status': 'tracking',
            'current_activity': self.current_activity,
            'current_gesture': self.current_gesture,
            'keypoints': keypoints,
            'tracking_stats': self._get_tracking_stats()
        }
    
    def _get_tracking_stats(self) -> Dict[str, Any]:
        """Get current tracking statistics."""
        return {
            'activities_detected': len(self.activity_history),
            'gestures_detected': len(self.gesture_history),
            'buffer_size': len(self.keypoint_buffer),
            'current_activity': self.current_activity.activity_type if self.current_activity else None,
            'current_gesture': self.current_gesture.gesture_type if self.current_gesture else None
        }
    
    def get_activity_summary(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Get activity summary for the specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        recent_activities = [a for a in self.activity_history if a.start_time >= cutoff_time]
        recent_gestures = [g for g in self.gesture_history if g.start_time >= cutoff_time]
        
        # Count activities
        activity_counts = {}
        for activity in recent_activities:
            activity_counts[activity.activity_type] = activity_counts.get(activity.activity_type, 0) + 1
        
        # Count gestures
        gesture_counts = {}
        for gesture in recent_gestures:
            gesture_counts[gesture.gesture_type] = gesture_counts.get(gesture.gesture_type, 0) + 1
        
        return {
            'duration_minutes': duration_minutes,
            'total_activities': len(recent_activities),
            'total_gestures': len(recent_gestures),
            'activity_counts': activity_counts,
            'gesture_counts': gesture_counts,
            'most_common_activity': max(activity_counts.items(), key=lambda x: x[1]) if activity_counts else None,
            'most_common_gesture': max(gesture_counts.items(), key=lambda x: x[1]) if gesture_counts else None
        }
