"""
Sports analysis and performance tracking application.
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import deque
from dataclasses import dataclass
import json

from ..inference import PoseDetector
from ..inference.utils import calculate_angles, calculate_distances, normalize_pose


@dataclass
class SportsEvent:
    """Data class for a sports event."""
    event_type: str
    start_time: float
    end_time: float
    keypoints_sequence: List[List[Dict[str, Any]]]
    performance_metrics: Dict[str, float]
    quality_score: float
    feedback: List[str]


@dataclass
class PerformanceAnalysis:
    """Data class for performance analysis."""
    session_id: str
    sport_type: str
    start_time: float
    end_time: float
    events: List[SportsEvent]
    overall_performance: float
    improvement_areas: List[str]
    strengths: List[str]
    recommendations: List[str]


class SportsAnalyzer:
    """Analyzes sports performance and technique."""
    
    def __init__(self):
        """Initialize sports analyzer."""
        # Sport-specific analysis configurations
        self.sport_configs = {
            'tennis': {
                'serve': {
                    'key_phases': ['preparation', 'backswing', 'contact', 'follow_through'],
                    'key_angles': ['shoulder_angle', 'elbow_angle', 'wrist_angle'],
                    'performance_metrics': ['power', 'accuracy', 'consistency']
                },
                'forehand': {
                    'key_phases': ['preparation', 'backswing', 'contact', 'follow_through'],
                    'key_angles': ['shoulder_angle', 'elbow_angle', 'wrist_angle'],
                    'performance_metrics': ['power', 'spin', 'placement']
                },
                'backhand': {
                    'key_phases': ['preparation', 'backswing', 'contact', 'follow_through'],
                    'key_angles': ['shoulder_angle', 'elbow_angle', 'wrist_angle'],
                    'performance_metrics': ['power', 'control', 'consistency']
                }
            },
            'basketball': {
                'shooting': {
                    'key_phases': ['setup', 'aim', 'release', 'follow_through'],
                    'key_angles': ['elbow_angle', 'wrist_angle', 'shoulder_angle'],
                    'performance_metrics': ['accuracy', 'arc', 'power']
                },
                'dribbling': {
                    'key_phases': ['control', 'bounce', 'protection'],
                    'key_angles': ['elbow_angle', 'wrist_angle'],
                    'performance_metrics': ['control', 'speed', 'protection']
                }
            },
            'golf': {
                'swing': {
                    'key_phases': ['address', 'backswing', 'downswing', 'impact', 'follow_through'],
                    'key_angles': ['shoulder_angle', 'hip_angle', 'knee_angle'],
                    'performance_metrics': ['power', 'accuracy', 'consistency']
                }
            },
            'swimming': {
                'freestyle': {
                    'key_phases': ['entry', 'catch', 'pull', 'push', 'recovery'],
                    'key_angles': ['shoulder_angle', 'elbow_angle', 'hip_angle'],
                    'performance_metrics': ['efficiency', 'power', 'rhythm']
                },
                'butterfly': {
                    'key_phases': ['entry', 'catch', 'pull', 'push', 'recovery'],
                    'key_angles': ['shoulder_angle', 'hip_angle', 'knee_angle'],
                    'performance_metrics': ['efficiency', 'power', 'coordination']
                }
            }
        }
    
    def analyze_sports_event(
        self, 
        sport_type: str, 
        event_type: str, 
        keypoints_sequence: List[List[Dict[str, Any]]]
    ) -> SportsEvent:
        """
        Analyze a sports event.
        
        Args:
            sport_type: Type of sport
            event_type: Type of event/move
            keypoints_sequence: Sequence of keypoints
        
        Returns:
            SportsEvent object with analysis results
        """
        if sport_type not in self.sport_configs:
            raise ValueError(f"Unsupported sport: {sport_type}")
        
        if event_type not in self.sport_configs[sport_type]:
            raise ValueError(f"Unsupported event type: {event_type}")
        
        config = self.sport_configs[sport_type][event_type]
        
        # Analyze phases
        phases = self._analyze_phases(keypoints_sequence, config['key_phases'])
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            keypoints_sequence, config['performance_metrics']
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            keypoints_sequence, config, performance_metrics
        )
        
        # Generate feedback
        feedback = self._generate_feedback(
            sport_type, event_type, performance_metrics, quality_score
        )
        
        return SportsEvent(
            event_type=event_type,
            start_time=time.time() - len(keypoints_sequence) * 0.1,
            end_time=time.time(),
            keypoints_sequence=keypoints_sequence,
            performance_metrics=performance_metrics,
            quality_score=quality_score,
            feedback=feedback
        )
    
    def _analyze_phases(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]], 
        key_phases: List[str]
    ) -> Dict[str, Any]:
        """Analyze movement phases."""
        if len(keypoints_sequence) < 4:
            return {}
        
        # Divide sequence into phases
        phase_length = len(keypoints_sequence) // len(key_phases)
        phases = {}
        
        for i, phase_name in enumerate(key_phases):
            start_idx = i * phase_length
            end_idx = (i + 1) * phase_length if i < len(key_phases) - 1 else len(keypoints_sequence)
            
            phase_keypoints = keypoints_sequence[start_idx:end_idx]
            phases[phase_name] = {
                'keypoints': phase_keypoints,
                'duration': (end_idx - start_idx) * 0.1,  # Assume 10fps
                'start_frame': start_idx,
                'end_frame': end_idx
            }
        
        return phases
    
    def _calculate_performance_metrics(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]], 
        metrics: List[str]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        performance_metrics = {}
        
        for metric in metrics:
            if metric == 'power':
                performance_metrics[metric] = self._calculate_power(keypoints_sequence)
            elif metric == 'accuracy':
                performance_metrics[metric] = self._calculate_accuracy(keypoints_sequence)
            elif metric == 'consistency':
                performance_metrics[metric] = self._calculate_consistency(keypoints_sequence)
            elif metric == 'efficiency':
                performance_metrics[metric] = self._calculate_efficiency(keypoints_sequence)
            elif metric == 'control':
                performance_metrics[metric] = self._calculate_control(keypoints_sequence)
            else:
                performance_metrics[metric] = 0.5  # Default value
        
        return performance_metrics
    
    def _calculate_power(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate power/force of movement."""
        if len(keypoints_sequence) < 3:
            return 0.0
        
        # Calculate velocity and acceleration
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
        
        if not velocities:
            return 0.0
        
        # Power is related to maximum velocity
        max_velocity = max(velocities)
        
        # Normalize to 0-1 scale
        power_score = min(1.0, max_velocity / 50.0)  # Adjust threshold as needed
        
        return power_score
    
    def _calculate_accuracy(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate movement accuracy."""
        if not keypoints_sequence:
            return 0.0
        
        # Calculate consistency in keypoint positions
        keypoint_variances = []
        
        for kp_idx in range(len(keypoints_sequence[0])):
            positions = []
            
            for frame_keypoints in keypoints_sequence:
                if (kp_idx < len(frame_keypoints) and 
                    frame_keypoints[kp_idx].get('confidence', 0) > 0.5):
                    positions.append([
                        frame_keypoints[kp_idx]['x'],
                        frame_keypoints[kp_idx]['y']
                    ])
            
            if len(positions) > 1:
                positions = np.array(positions)
                variance = np.var(positions, axis=0).mean()
                keypoint_variances.append(variance)
        
        if not keypoint_variances:
            return 0.0
        
        # Lower variance = higher accuracy
        avg_variance = np.mean(keypoint_variances)
        accuracy_score = max(0, 1 - avg_variance / 100.0)  # Adjust threshold
        
        return accuracy_score
    
    def _calculate_consistency(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate movement consistency."""
        if len(keypoints_sequence) < 3:
            return 0.0
        
        # Calculate consistency by comparing movement patterns
        movement_patterns = []
        
        for i in range(len(keypoints_sequence) - 1):
            prev_kp = keypoints_sequence[i]
            curr_kp = keypoints_sequence[i + 1]
            
            # Calculate movement vector
            movement = []
            for j in range(min(len(prev_kp), len(curr_kp))):
                if (prev_kp[j].get('confidence', 0) > 0.5 and 
                    curr_kp[j].get('confidence', 0) > 0.5):
                    
                    dx = curr_kp[j]['x'] - prev_kp[j]['x']
                    dy = curr_kp[j]['y'] - prev_kp[j]['y']
                    movement.extend([dx, dy])
            
            if movement:
                movement_patterns.append(movement)
        
        if len(movement_patterns) < 2:
            return 0.0
        
        # Calculate pattern similarity
        similarities = []
        for i in range(len(movement_patterns) - 1):
            pattern1 = np.array(movement_patterns[i])
            pattern2 = np.array(movement_patterns[i + 1])
            
            if len(pattern1) == len(pattern2) and len(pattern1) > 0:
                # Calculate cosine similarity
                similarity = np.dot(pattern1, pattern2) / (
                    np.linalg.norm(pattern1) * np.linalg.norm(pattern2)
                )
                similarities.append(similarity)
        
        if similarities:
            consistency_score = np.mean(similarities)
            return max(0, consistency_score)
        
        return 0.0
    
    def _calculate_efficiency(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate movement efficiency."""
        if not keypoints_sequence:
            return 0.0
        
        # Efficiency is related to smoothness and economy of movement
        smoothness = self._calculate_smoothness(keypoints_sequence)
        economy = self._calculate_economy(keypoints_sequence)
        
        efficiency_score = (smoothness + economy) / 2
        return efficiency_score
    
    def _calculate_smoothness(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate movement smoothness."""
        if len(keypoints_sequence) < 3:
            return 0.0
        
        # Calculate jerk (third derivative of position)
        jerks = []
        
        for i in range(2, len(keypoints_sequence)):
            prev_prev_kp = keypoints_sequence[i-2]
            prev_kp = keypoints_sequence[i-1]
            curr_kp = keypoints_sequence[i]
            
            jerk_sum = 0
            valid_points = 0
            
            for j in range(min(len(prev_prev_kp), len(prev_kp), len(curr_kp))):
                if all(kp[j].get('confidence', 0) > 0.5 for kp in [prev_prev_kp, prev_kp, curr_kp]):
                    # Calculate jerk
                    pos1 = [prev_prev_kp[j]['x'], prev_prev_kp[j]['y']]
                    pos2 = [prev_kp[j]['x'], prev_kp[j]['y']]
                    pos3 = [curr_kp[j]['x'], curr_kp[j]['y']]
                    
                    # Simplified jerk calculation
                    jerk = abs(pos3[0] - 2*pos2[0] + pos1[0]) + abs(pos3[1] - 2*pos2[1] + pos1[1])
                    jerk_sum += jerk
                    valid_points += 1
            
            if valid_points > 0:
                jerks.append(jerk_sum / valid_points)
        
        if jerks:
            avg_jerk = np.mean(jerks)
            smoothness_score = max(0, 1 - avg_jerk / 10.0)  # Adjust threshold
            return smoothness_score
        
        return 0.0
    
    def _calculate_economy(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate movement economy."""
        if not keypoints_sequence:
            return 0.0
        
        # Economy is related to minimizing unnecessary movements
        total_movement = 0
        valid_frames = 0
        
        for i in range(1, len(keypoints_sequence)):
            prev_kp = keypoints_sequence[i-1]
            curr_kp = keypoints_sequence[i]
            
            frame_movement = 0
            valid_points = 0
            
            for j in range(min(len(prev_kp), len(curr_kp))):
                if (prev_kp[j].get('confidence', 0) > 0.5 and 
                    curr_kp[j].get('confidence', 0) > 0.5):
                    
                    dx = curr_kp[j]['x'] - prev_kp[j]['x']
                    dy = curr_kp[j]['y'] - prev_kp[j]['y']
                    movement = np.sqrt(dx**2 + dy**2)
                    
                    frame_movement += movement
                    valid_points += 1
            
            if valid_points > 0:
                total_movement += frame_movement / valid_points
                valid_frames += 1
        
        if valid_frames == 0:
            return 0.0
        
        # Lower total movement = higher economy
        avg_movement = total_movement / valid_frames
        economy_score = max(0, 1 - avg_movement / 20.0)  # Adjust threshold
        
        return economy_score
    
    def _calculate_control(self, keypoints_sequence: List[List[Dict[str, Any]]]) -> float:
        """Calculate movement control."""
        # Control is combination of accuracy and consistency
        accuracy = self._calculate_accuracy(keypoints_sequence)
        consistency = self._calculate_consistency(keypoints_sequence)
        
        control_score = (accuracy + consistency) / 2
        return control_score
    
    def _calculate_quality_score(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]], 
        config: Dict[str, Any], 
        performance_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall quality score."""
        # Weight different metrics based on sport and event
        weights = {
            'power': 0.3,
            'accuracy': 0.4,
            'consistency': 0.3,
            'efficiency': 0.4,
            'control': 0.3
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in performance_metrics.items():
            if metric in weights:
                weighted_score += score * weights[metric]
                total_weight += weights[metric]
        
        if total_weight > 0:
            quality_score = weighted_score / total_weight
        else:
            quality_score = 0.5
        
        return quality_score
    
    def _generate_feedback(
        self, 
        sport_type: str, 
        event_type: str, 
        performance_metrics: Dict[str, float], 
        quality_score: float
    ) -> List[str]:
        """Generate feedback based on performance."""
        feedback = []
        
        # Overall feedback
        if quality_score >= 0.8:
            feedback.append("Excellent technique! Keep up the great work!")
        elif quality_score >= 0.6:
            feedback.append("Good technique with room for improvement")
        else:
            feedback.append("Focus on fundamental technique")
        
        # Specific feedback based on metrics
        for metric, score in performance_metrics.items():
            if score < 0.5:
                if metric == 'power':
                    feedback.append("Work on generating more power through proper technique")
                elif metric == 'accuracy':
                    feedback.append("Focus on improving accuracy and precision")
                elif metric == 'consistency':
                    feedback.append("Practice for more consistent movement patterns")
                elif metric == 'efficiency':
                    feedback.append("Work on making movements more efficient")
                elif metric == 'control':
                    feedback.append("Improve control and stability")
        
        return feedback


class PerformanceTracker:
    """Tracks sports performance over time."""
    
    def __init__(self, model_type: str = 'mediapipe'):
        """
        Initialize performance tracker.
        
        Args:
            model_type: Type of pose estimation model to use
        """
        self.detector = PoseDetector(model_type=model_type)
        self.analyzer = SportsAnalyzer()
        
        # Tracking state
        self.current_session = None
        self.event_buffer = deque(maxlen=30)  # Buffer for recent events
        self.is_tracking = False
        
        # Performance history
        self.performance_history = []
    
    def start_tracking_session(
        self, 
        sport_type: str, 
        session_name: str = None
    ) -> PerformanceAnalysis:
        """
        Start a new performance tracking session.
        
        Args:
            sport_type: Type of sport
            session_name: Name for the session
        
        Returns:
            PerformanceAnalysis object
        """
        session_id = f"{sport_type}_{int(time.time())}"
        
        self.current_session = PerformanceAnalysis(
            session_id=session_id,
            sport_type=sport_type,
            start_time=time.time(),
            end_time=0.0,
            events=[],
            overall_performance=0.0,
            improvement_areas=[],
            strengths=[],
            recommendations=[]
        )
        
        self.is_tracking = True
        print(f"Started {sport_type} performance tracking session")
        
        return self.current_session
    
    def stop_tracking_session(self) -> PerformanceAnalysis:
        """
        Stop the current tracking session.
        
        Returns:
            Completed PerformanceAnalysis object
        """
        if not self.current_session:
            raise ValueError("No active tracking session")
        
        self.current_session.end_time = time.time()
        
        # Calculate overall performance
        if self.current_session.events:
            quality_scores = [event.quality_score for event in self.current_session.events]
            self.current_session.overall_performance = np.mean(quality_scores)
            
            # Analyze strengths and improvement areas
            self.current_session.strengths = self._identify_strengths(self.current_session.events)
            self.current_session.improvement_areas = self._identify_improvement_areas(self.current_session.events)
            self.current_session.recommendations = self._generate_recommendations(self.current_session)
        
        # Reset tracking state
        self.is_tracking = False
        session = self.current_session
        self.current_session = None
        
        # Add to history
        self.performance_history.append(session)
        
        print(f"Completed {session.sport_type} performance tracking session")
        print(f"Overall performance: {session.overall_performance:.2f}")
        print(f"Events analyzed: {len(session.events)}")
        
        return session
    
    def analyze_event(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]], 
        event_type: str = None
    ) -> SportsEvent:
        """
        Analyze a sports event.
        
        Args:
            keypoints_sequence: Sequence of keypoints for the event
            event_type: Type of event (if not provided, will be auto-detected)
        
        Returns:
            SportsEvent object
        """
        if not self.current_session:
            raise ValueError("No active tracking session")
        
        sport_type = self.current_session.sport_type
        
        # Auto-detect event type if not provided
        if event_type is None:
            event_type = self._detect_event_type(keypoints_sequence, sport_type)
        
        # Analyze the event
        event = self.analyzer.analyze_sports_event(sport_type, event_type, keypoints_sequence)
        
        # Add to current session
        self.current_session.events.append(event)
        
        return event
    
    def _detect_event_type(
        self, 
        keypoints_sequence: List[List[Dict[str, Any]]], 
        sport_type: str
    ) -> str:
        """Auto-detect event type from keypoints sequence."""
        # Simplified event detection
        # In practice, you'd use more sophisticated methods
        
        if sport_type == 'tennis':
            # Analyze movement patterns to detect serve, forehand, backhand
            return 'serve'  # Default for now
        elif sport_type == 'basketball':
            return 'shooting'  # Default for now
        elif sport_type == 'golf':
            return 'swing'  # Default for now
        else:
            return 'general'
    
    def _identify_strengths(self, events: List[SportsEvent]) -> List[str]:
        """Identify performance strengths."""
        strengths = []
        
        if not events:
            return strengths
        
        # Analyze metrics across all events
        all_metrics = {}
        for event in events:
            for metric, score in event.performance_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(score)
        
        # Identify strong areas (average score > 0.7)
        for metric, scores in all_metrics.items():
            avg_score = np.mean(scores)
            if avg_score > 0.7:
                strengths.append(f"Strong {metric} (score: {avg_score:.2f})")
        
        return strengths
    
    def _identify_improvement_areas(self, events: List[SportsEvent]) -> List[str]:
        """Identify areas for improvement."""
        improvement_areas = []
        
        if not events:
            return improvement_areas
        
        # Analyze metrics across all events
        all_metrics = {}
        for event in events:
            for metric, score in event.performance_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(score)
        
        # Identify weak areas (average score < 0.5)
        for metric, scores in all_metrics.items():
            avg_score = np.mean(scores)
            if avg_score < 0.5:
                improvement_areas.append(f"Improve {metric} (current score: {avg_score:.2f})")
        
        return improvement_areas
    
    def _generate_recommendations(self, session: PerformanceAnalysis) -> List[str]:
        """Generate training recommendations."""
        recommendations = []
        
        # General recommendations based on overall performance
        if session.overall_performance < 0.5:
            recommendations.append("Focus on fundamental technique and basic movements")
        elif session.overall_performance < 0.7:
            recommendations.append("Continue practicing with emphasis on consistency")
        else:
            recommendations.append("Excellent performance! Work on fine-tuning and advanced techniques")
        
        # Specific recommendations based on improvement areas
        for area in session.improvement_areas:
            if 'power' in area.lower():
                recommendations.append("Practice explosive movements and proper sequencing")
            elif 'accuracy' in area.lower():
                recommendations.append("Focus on precision training and target practice")
            elif 'consistency' in area.lower():
                recommendations.append("Repetitive practice to develop muscle memory")
            elif 'efficiency' in area.lower():
                recommendations.append("Work on economy of movement and energy conservation")
        
        return recommendations
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        recent_sessions = [s for s in self.performance_history if s.start_time >= cutoff_time]
        
        if not recent_sessions:
            return {'trend': 'no_data', 'sessions': 0}
        
        # Calculate trend
        performances = [s.overall_performance for s in recent_sessions]
        
        if len(performances) >= 2:
            # Simple trend calculation
            early_avg = np.mean(performances[:len(performances)//2])
            late_avg = np.mean(performances[len(performances)//2:])
            
            if late_avg > early_avg + 0.1:
                trend = 'improving'
            elif late_avg < early_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'sessions': len(recent_sessions),
            'average_performance': np.mean(performances),
            'best_performance': np.max(performances),
            'recent_performances': performances[-5:]  # Last 5 sessions
        }
