"""
Fitness tracking and exercise form correction application.
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import deque
from dataclasses import dataclass

from ..inference import PoseDetector
from ..inference.utils import calculate_angles, calculate_distances, normalize_pose


@dataclass
class ExerciseRep:
    """Data class for a single exercise repetition."""
    start_time: float
    end_time: float
    keypoints_sequence: List[List[Dict[str, Any]]]
    angles_sequence: List[Dict[str, float]]
    quality_score: float
    feedback: List[str]


@dataclass
class ExerciseSession:
    """Data class for an exercise session."""
    exercise_name: str
    start_time: float
    end_time: float
    reps: List[ExerciseRep]
    total_reps: int
    average_quality: float
    calories_burned: float
    summary: Dict[str, Any]


class ExerciseAnalyzer:
    """Analyzes exercise form and provides feedback."""
    
    def __init__(self):
        """Initialize exercise analyzer."""
        # Exercise-specific configurations
        self.exercise_configs = {
            'push_up': {
                'key_angles': ['left_elbow_angle', 'right_elbow_angle'],
                'key_distances': ['shoulder_width', 'hip_width'],
                'rep_phases': ['down', 'up'],
                'quality_thresholds': {
                    'elbow_angle_min': 80,
                    'elbow_angle_max': 180,
                    'back_straightness': 0.8
                }
            },
            'squat': {
                'key_angles': ['left_knee_angle', 'right_knee_angle', 'hip_angle'],
                'key_distances': ['knee_width'],
                'rep_phases': ['down', 'up'],
                'quality_thresholds': {
                    'knee_angle_min': 70,
                    'hip_angle_min': 90,
                    'knee_tracking': 0.7
                }
            },
            'pull_up': {
                'key_angles': ['left_elbow_angle', 'right_elbow_angle'],
                'key_distances': ['shoulder_width'],
                'rep_phases': ['down', 'up'],
                'quality_thresholds': {
                    'elbow_angle_min': 30,
                    'elbow_angle_max': 180,
                    'shoulder_stability': 0.8
                }
            },
            'plank': {
                'key_angles': ['hip_angle', 'shoulder_angle'],
                'key_distances': ['body_length'],
                'rep_phases': ['hold'],
                'quality_thresholds': {
                    'hip_angle_min': 170,
                    'hip_angle_max': 190,
                    'shoulder_stability': 0.9
                }
            }
        }
    
    def analyze_exercise(
        self, 
        exercise_name: str, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> ExerciseRep:
        """
        Analyze a single exercise repetition.
        
        Args:
            exercise_name: Name of the exercise
            keypoints_sequence: Sequence of keypoints for the rep
        
        Returns:
            ExerciseRep object with analysis results
        """
        if exercise_name not in self.exercise_configs:
            raise ValueError(f"Unknown exercise: {exercise_name}")
        
        config = self.exercise_configs[exercise_name]
        
        # Calculate angles for each frame
        angles_sequence = []
        for keypoints in keypoints_sequence:
            angles = self._calculate_exercise_angles(keypoints, exercise_name)
            angles_sequence.append(angles)
        
        # Analyze form quality
        quality_score, feedback = self._analyze_form_quality(
            keypoints_sequence, angles_sequence, config
        )
        
        # Create ExerciseRep object
        rep = ExerciseRep(
            start_time=0.0,  # Will be set by caller
            end_time=0.0,    # Will be set by caller
            keypoints_sequence=keypoints_sequence,
            angles_sequence=angles_sequence,
            quality_score=quality_score,
            feedback=feedback
        )
        
        return rep
    
    def _calculate_exercise_angles(
        self, 
        keypoints: List[Dict[str, Any]], 
        exercise_name: str
    ) -> Dict[str, float]:
        """Calculate exercise-specific angles."""
        angle_configs = []
        
        if exercise_name == 'push_up':
            # Elbow angles
            angle_configs.extend([
                {'name': 'left_elbow_angle', 'point1': 5, 'vertex': 7, 'point2': 9},   # left shoulder-elbow-wrist
                {'name': 'right_elbow_angle', 'point1': 6, 'vertex': 8, 'point2': 10}, # right shoulder-elbow-wrist
            ])
        elif exercise_name == 'squat':
            # Knee and hip angles
            angle_configs.extend([
                {'name': 'left_knee_angle', 'point1': 11, 'vertex': 13, 'point2': 15},  # left hip-knee-ankle
                {'name': 'right_knee_angle', 'point1': 12, 'vertex': 14, 'point2': 16}, # right hip-knee-ankle
                {'name': 'hip_angle', 'point1': 5, 'vertex': 11, 'point2': 6},          # shoulder-hip-shoulder
            ])
        elif exercise_name == 'pull_up':
            # Elbow angles
            angle_configs.extend([
                {'name': 'left_elbow_angle', 'point1': 5, 'vertex': 7, 'point2': 9},   # left shoulder-elbow-wrist
                {'name': 'right_elbow_angle', 'point1': 6, 'vertex': 8, 'point2': 10}, # right shoulder-elbow-wrist
            ])
        elif exercise_name == 'plank':
            # Hip and shoulder angles
            angle_configs.extend([
                {'name': 'hip_angle', 'point1': 5, 'vertex': 11, 'point2': 6},         # shoulder-hip-shoulder
                {'name': 'shoulder_angle', 'point1': 11, 'vertex': 5, 'point2': 7},    # hip-shoulder-elbow
            ])
        
        return calculate_angles(keypoints, angle_configs)
    
    def _analyze_form_quality(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]],
        angles_sequence: List[Dict[str, float]],
        config: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Analyze exercise form quality."""
        feedback = []
        quality_factors = []
        
        # Analyze each quality factor
        for angle_name in config['key_angles']:
            if angle_name in angles_sequence[0]:
                angle_values = [angles[angle_name] for angles in angles_sequence 
                              if angles[angle_name] is not None]
                
                if angle_values:
                    avg_angle = np.mean(angle_values)
                    min_angle = np.min(angle_values)
                    max_angle = np.max(angle_values)
                    
                    # Check angle thresholds
                    if angle_name == 'left_elbow_angle' or angle_name == 'right_elbow_angle':
                        if 'elbow_angle_min' in config['quality_thresholds']:
                            if min_angle < config['quality_thresholds']['elbow_angle_min']:
                                feedback.append(f"Elbow angle too small: {min_angle:.1f}°")
                                quality_factors.append(0.5)
                            else:
                                quality_factors.append(1.0)
                    
                    elif angle_name == 'left_knee_angle' or angle_name == 'right_knee_angle':
                        if 'knee_angle_min' in config['quality_thresholds']:
                            if min_angle < config['quality_thresholds']['knee_angle_min']:
                                feedback.append(f"Knee angle too small: {min_angle:.1f}°")
                                quality_factors.append(0.5)
                            else:
                                quality_factors.append(1.0)
        
        # Analyze movement consistency
        consistency_score = self._analyze_movement_consistency(keypoints_sequence)
        quality_factors.append(consistency_score)
        
        if consistency_score < 0.7:
            feedback.append("Movement not smooth and consistent")
        
        # Calculate overall quality score
        if quality_factors:
            overall_quality = np.mean(quality_factors)
        else:
            overall_quality = 0.5
        
        # Add positive feedback for good form
        if overall_quality > 0.8:
            feedback.append("Great form! Keep it up!")
        elif overall_quality > 0.6:
            feedback.append("Good form with minor improvements needed")
        else:
            feedback.append("Focus on proper form and technique")
        
        return overall_quality, feedback
    
    def _analyze_movement_consistency(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> float:
        """Analyze movement consistency throughout the rep."""
        if len(keypoints_sequence) < 3:
            return 0.5
        
        # Calculate velocity consistency
        velocities = []
        for i in range(1, len(keypoints_sequence)):
            prev_kp = keypoints_sequence[i-1]
            curr_kp = keypoints_sequence[i]
            
            # Calculate average velocity for visible keypoints
            velocity_sum = 0
            valid_points = 0
            
            for j in range(min(len(prev_kp), len(curr_kp))):
                if (prev_kp[j].get('confidence', 0) > 0.5 and 
                    curr_kp[j].get('confidence', 0) > 0.5):
                    
                    dx = curr_kp[j]['x'] - prev_kp[j]['x']
                    dy = curr_kp[j]['y'] - prev_kp[j]['y']
                    velocity = np.sqrt(dx**2 + dy**2)
                    
                    velocity_sum += velocity
                    valid_points += 1
            
            if valid_points > 0:
                avg_velocity = velocity_sum / valid_points
                velocities.append(avg_velocity)
        
        if len(velocities) < 2:
            return 0.5
        
        # Calculate velocity variance (lower is more consistent)
        velocity_variance = np.var(velocities)
        
        # Convert to consistency score (0-1)
        consistency_score = max(0, 1 - velocity_variance / 100)
        
        return consistency_score


class FitnessTracker:
    """Main fitness tracking application."""
    
    def __init__(self, model_type: str = 'mediapipe'):
        """
        Initialize fitness tracker.
        
        Args:
            model_type: Type of pose estimation model to use
        """
        self.detector = PoseDetector(model_type=model_type)
        self.analyzer = ExerciseAnalyzer()
        
        # Tracking state
        self.current_exercise = None
        self.current_session = None
        self.keypoint_buffer = deque(maxlen=30)  # Buffer for recent keypoints
        self.is_tracking = False
        
        # Exercise detection
        self.exercise_detector = ExerciseDetector()
        
        # Calorie estimation
        self.calorie_estimator = CalorieEstimator()
    
    def start_exercise_session(self, exercise_name: str) -> ExerciseSession:
        """
        Start a new exercise session.
        
        Args:
            exercise_name: Name of the exercise
        
        Returns:
            ExerciseSession object
        """
        self.current_exercise = exercise_name
        self.current_session = ExerciseSession(
            exercise_name=exercise_name,
            start_time=time.time(),
            end_time=0.0,
            reps=[],
            total_reps=0,
            average_quality=0.0,
            calories_burned=0.0,
            summary={}
        )
        self.is_tracking = True
        
        print(f"Started {exercise_name} session")
        return self.current_session
    
    def stop_exercise_session(self) -> ExerciseSession:
        """
        Stop the current exercise session.
        
        Returns:
            Completed ExerciseSession object
        """
        if not self.current_session:
            raise ValueError("No active exercise session")
        
        self.current_session.end_time = time.time()
        
        # Calculate session statistics
        if self.current_session.reps:
            self.current_session.average_quality = np.mean([
                rep.quality_score for rep in self.current_session.reps
            ])
            self.current_session.calories_burned = self.calorie_estimator.estimate_calories(
                self.current_session
            )
        
        self.current_session.summary = self._generate_session_summary(self.current_session)
        
        # Reset tracking state
        self.is_tracking = False
        self.current_exercise = None
        self.keypoint_buffer.clear()
        
        print(f"Completed {self.current_session.exercise_name} session")
        print(f"Total reps: {self.current_session.total_reps}")
        print(f"Average quality: {self.current_session.average_quality:.2f}")
        print(f"Calories burned: {self.current_session.calories_burned:.1f}")
        
        return self.current_session
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for exercise tracking.
        
        Args:
            frame: Input video frame
        
        Returns:
            Dictionary containing tracking results
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
        
        # Detect exercise phases and reps
        rep_detection = self._detect_rep(keypoints)
        
        # Analyze current form if we have enough frames
        form_analysis = None
        if len(self.keypoint_buffer) >= 10:
            recent_keypoints = list(self.keypoint_buffer)[-10:]
            angles = self._calculate_current_angles(recent_keypoints)
            form_analysis = self._analyze_current_form(angles)
        
        return {
            'status': 'tracking',
            'keypoints': keypoints,
            'rep_detection': rep_detection,
            'form_analysis': form_analysis,
            'session_stats': self._get_session_stats()
        }
    
    def _detect_rep(self, keypoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect exercise repetition phases."""
        if not self.current_exercise:
            return {'phase': 'unknown'}
        
        # Calculate current angles
        angles = self._calculate_current_angles([keypoints])
        if not angles:
            return {'phase': 'unknown'}
        
        current_angles = angles[0]
        
        # Detect phase based on exercise type
        if self.current_exercise == 'push_up':
            elbow_angles = [current_angles.get('left_elbow_angle', 0),
                          current_angles.get('right_elbow_angle', 0)]
            avg_elbow_angle = np.mean([a for a in elbow_angles if a is not None])
            
            if avg_elbow_angle < 90:
                phase = 'down'
            else:
                phase = 'up'
        
        elif self.current_exercise == 'squat':
            knee_angles = [current_angles.get('left_knee_angle', 0),
                          current_angles.get('right_knee_angle', 0)]
            avg_knee_angle = np.mean([a for a in knee_angles if a is not None])
            
            if avg_knee_angle < 110:
                phase = 'down'
            else:
                phase = 'up'
        
        else:
            phase = 'unknown'
        
        # Check for rep completion
        rep_completed = self._check_rep_completion(phase)
        
        return {
            'phase': phase,
            'rep_completed': rep_completed,
            'angles': current_angles
        }
    
    def _check_rep_completion(self, current_phase: str) -> bool:
        """Check if a repetition has been completed."""
        # This is a simplified version - in practice, you'd track phase transitions
        # and detect complete up-down-up or down-up-down cycles
        
        # For now, just return False
        # In a real implementation, you'd track the previous phase and detect transitions
        return False
    
    def _calculate_current_angles(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> List[Dict[str, float]]:
        """Calculate angles for current keypoints."""
        if not self.current_exercise:
            return []
        
        angles_sequence = []
        for keypoints in keypoints_sequence:
            angles = self.analyzer._calculate_exercise_angles(keypoints, self.current_exercise)
            angles_sequence.append(angles)
        
        return angles_sequence
    
    def _analyze_current_form(self, angles_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze current form based on recent angles."""
        if not angles_sequence:
            return {}
        
        current_angles = angles_sequence[-1]
        
        # Simple form analysis
        form_score = 1.0
        feedback = []
        
        if self.current_exercise == 'push_up':
            elbow_angles = [current_angles.get('left_elbow_angle', 0),
                          current_angles.get('right_elbow_angle', 0)]
            avg_elbow_angle = np.mean([a for a in elbow_angles if a is not None])
            
            if avg_elbow_angle < 80:
                feedback.append("Go lower for better range of motion")
                form_score -= 0.2
            elif avg_elbow_angle > 160:
                feedback.append("Don't lock your elbows")
                form_score -= 0.1
        
        return {
            'form_score': max(0, form_score),
            'feedback': feedback,
            'angles': current_angles
        }
    
    def _get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        if not self.current_session:
            return {}
        
        return {
            'exercise_name': self.current_session.exercise_name,
            'total_reps': self.current_session.total_reps,
            'session_duration': time.time() - self.current_session.start_time,
            'average_quality': self.current_session.average_quality
        }
    
    def _generate_session_summary(self, session: ExerciseSession) -> Dict[str, Any]:
        """Generate summary of exercise session."""
        duration = session.end_time - session.start_time
        
        summary = {
            'duration_minutes': duration / 60,
            'total_reps': session.total_reps,
            'reps_per_minute': (session.total_reps / duration) * 60 if duration > 0 else 0,
            'average_quality': session.average_quality,
            'calories_burned': session.calories_burned,
            'quality_distribution': self._get_quality_distribution(session.reps)
        }
        
        return summary
    
    def _get_quality_distribution(self, reps: List[ExerciseRep]) -> Dict[str, int]:
        """Get distribution of quality scores."""
        if not reps:
            return {}
        
        excellent = sum(1 for rep in reps if rep.quality_score >= 0.8)
        good = sum(1 for rep in reps if 0.6 <= rep.quality_score < 0.8)
        fair = sum(1 for rep in reps if 0.4 <= rep.quality_score < 0.6)
        poor = sum(1 for rep in reps if rep.quality_score < 0.4)
        
        return {
            'excellent': excellent,
            'good': good,
            'fair': fair,
            'poor': poor
        }


class ExerciseDetector:
    """Detects exercise type from pose sequence."""
    
    def __init__(self):
        """Initialize exercise detector."""
        pass
    
    def detect_exercise(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> str:
        """
        Detect exercise type from keypoint sequence.
        
        Args:
            keypoints_sequence: Sequence of keypoints
        
        Returns:
            Detected exercise name
        """
        # Simplified exercise detection
        # In practice, you'd use machine learning models or rule-based systems
        
        if not keypoints_sequence:
            return 'unknown'
        
        # Analyze movement patterns
        movement_pattern = self._analyze_movement_pattern(keypoints_sequence)
        
        # Classify based on patterns
        if movement_pattern['vertical_movement'] > 0.5:
            if movement_pattern['elbow_bend'] > 0.3:
                return 'push_up'
            else:
                return 'squat'
        elif movement_pattern['horizontal_movement'] > 0.5:
            return 'plank'
        else:
            return 'unknown'
    
    def _analyze_movement_pattern(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Analyze movement patterns in keypoint sequence."""
        # Simplified pattern analysis
        return {
            'vertical_movement': 0.5,
            'horizontal_movement': 0.3,
            'elbow_bend': 0.4
        }


class CalorieEstimator:
    """Estimates calories burned during exercise."""
    
    def __init__(self):
        """Initialize calorie estimator."""
        # Base metabolic rates for different exercises (calories per minute)
        self.exercise_rates = {
            'push_up': 8.0,
            'squat': 7.5,
            'pull_up': 9.0,
            'plank': 4.0
        }
    
    def estimate_calories(self, session: ExerciseSession) -> float:
        """
        Estimate calories burned during exercise session.
        
        Args:
            session: Exercise session data
        
        Returns:
            Estimated calories burned
        """
        duration_minutes = (session.end_time - session.start_time) / 60
        
        base_rate = self.exercise_rates.get(session.exercise_name, 5.0)
        
        # Adjust for intensity based on quality and reps
        intensity_factor = 1.0
        if session.total_reps > 0:
            reps_per_minute = session.total_reps / duration_minutes
            intensity_factor = min(2.0, 1.0 + reps_per_minute / 10)
        
        # Adjust for form quality
        quality_factor = 0.8 + (session.average_quality * 0.4)
        
        calories = base_rate * duration_minutes * intensity_factor * quality_factor
        
        return calories
