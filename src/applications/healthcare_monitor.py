"""
Healthcare monitoring and rehabilitation tracking application.
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
class PostureAssessment:
    """Data class for posture assessment."""
    timestamp: float
    posture_score: float
    issues: List[str]
    recommendations: List[str]
    keypoints: List[Dict[str, Any]]


@dataclass
class RehabilitationSession:
    """Data class for rehabilitation session."""
    session_id: str
    patient_id: str
    exercise_type: str
    start_time: float
    end_time: float
    assessments: List[PostureAssessment]
    progress_score: float
    recommendations: List[str]


class PostureAnalyzer:
    """Analyzes posture and provides health recommendations."""
    
    def __init__(self):
        """Initialize posture analyzer."""
        # Posture analysis configurations
        self.posture_standards = {
            'standing': {
                'head_alignment': {'tolerance': 10, 'weight': 0.3},
                'shoulder_level': {'tolerance': 5, 'weight': 0.2},
                'hip_alignment': {'tolerance': 8, 'weight': 0.2},
                'spine_straightness': {'tolerance': 15, 'weight': 0.3}
            },
            'sitting': {
                'back_support': {'tolerance': 20, 'weight': 0.4},
                'leg_position': {'tolerance': 15, 'weight': 0.3},
                'head_position': {'tolerance': 12, 'weight': 0.3}
            },
            'walking': {
                'gait_symmetry': {'tolerance': 0.1, 'weight': 0.4},
                'stride_length': {'tolerance': 0.2, 'weight': 0.3},
                'posture_maintenance': {'tolerance': 15, 'weight': 0.3}
            }
        }
    
    def analyze_posture(
        self, 
        keypoints: List[Dict[str, Any]], 
        posture_type: str = 'standing'
    ) -> PostureAssessment:
        """
        Analyze posture from keypoints.
        
        Args:
            keypoints: List of keypoint dictionaries
            posture_type: Type of posture to analyze
        
        Returns:
            PostureAssessment object
        """
        if posture_type not in self.posture_standards:
            raise ValueError(f"Unknown posture type: {posture_type}")
        
        standards = self.posture_standards[posture_type]
        issues = []
        recommendations = []
        
        # Analyze different aspects of posture
        if posture_type == 'standing':
            posture_score, issues, recommendations = self._analyze_standing_posture(
                keypoints, standards
            )
        elif posture_type == 'sitting':
            posture_score, issues, recommendations = self._analyze_sitting_posture(
                keypoints, standards
            )
        elif posture_type == 'walking':
            posture_score, issues, recommendations = self._analyze_walking_posture(
                keypoints, standards
            )
        else:
            posture_score = 0.5
            issues = ["Unknown posture type"]
            recommendations = ["Please specify correct posture type"]
        
        return PostureAssessment(
            timestamp=time.time(),
            posture_score=posture_score,
            issues=issues,
            recommendations=recommendations,
            keypoints=keypoints
        )
    
    def _analyze_standing_posture(
        self, 
        keypoints: List[Dict[str, Any]], 
        standards: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze standing posture."""
        issues = []
        recommendations = []
        scores = []
        
        # Head alignment analysis
        head_score, head_issues, head_recs = self._check_head_alignment(keypoints)
        scores.append(head_score * standards['head_alignment']['weight'])
        issues.extend(head_issues)
        recommendations.extend(head_recs)
        
        # Shoulder level analysis
        shoulder_score, shoulder_issues, shoulder_recs = self._check_shoulder_level(keypoints)
        scores.append(shoulder_score * standards['shoulder_level']['weight'])
        issues.extend(shoulder_issues)
        recommendations.extend(shoulder_recs)
        
        # Hip alignment analysis
        hip_score, hip_issues, hip_recs = self._check_hip_alignment(keypoints)
        scores.append(hip_score * standards['hip_alignment']['weight'])
        issues.extend(hip_issues)
        recommendations.extend(hip_recs)
        
        # Spine straightness analysis
        spine_score, spine_issues, spine_recs = self._check_spine_straightness(keypoints)
        scores.append(spine_score * standards['spine_straightness']['weight'])
        issues.extend(spine_issues)
        recommendations.extend(spine_recs)
        
        overall_score = sum(scores)
        
        return overall_score, issues, recommendations
    
    def _check_head_alignment(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check head alignment."""
        issues = []
        recommendations = []
        
        # Get head and shoulder keypoints
        nose = self._get_keypoint(keypoints, 0)  # nose
        left_shoulder = self._get_keypoint(keypoints, 5)  # left shoulder
        right_shoulder = self._get_keypoint(keypoints, 6)  # right shoulder
        
        if not all([nose, left_shoulder, right_shoulder]):
            return 0.5, ["Insufficient keypoint data"], ["Ensure good lighting and clear view"]
        
        # Calculate shoulder center
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        
        # Check if head is centered over shoulders
        head_offset = abs(nose['x'] - shoulder_center_x)
        
        if head_offset > 20:  # pixels
            issues.append("Head not centered over shoulders")
            recommendations.append("Try to keep your head centered and balanced")
            score = max(0, 1 - head_offset / 50)
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _check_shoulder_level(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check shoulder level."""
        issues = []
        recommendations = []
        
        left_shoulder = self._get_keypoint(keypoints, 5)  # left shoulder
        right_shoulder = self._get_keypoint(keypoints, 6)  # right shoulder
        
        if not all([left_shoulder, right_shoulder]):
            return 0.5, ["Insufficient shoulder data"], ["Ensure shoulders are visible"]
        
        # Check shoulder height difference
        height_diff = abs(left_shoulder['y'] - right_shoulder['y'])
        
        if height_diff > 15:  # pixels
            issues.append("Uneven shoulder height")
            recommendations.append("Try to keep shoulders level and relaxed")
            score = max(0, 1 - height_diff / 30)
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _check_hip_alignment(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check hip alignment."""
        issues = []
        recommendations = []
        
        left_hip = self._get_keypoint(keypoints, 11)  # left hip
        right_hip = self._get_keypoint(keypoints, 12)  # right hip
        
        if not all([left_hip, right_hip]):
            return 0.5, ["Insufficient hip data"], ["Ensure hips are visible"]
        
        # Check hip height difference
        height_diff = abs(left_hip['y'] - right_hip['y'])
        
        if height_diff > 20:  # pixels
            issues.append("Uneven hip alignment")
            recommendations.append("Check for leg length discrepancy or muscle imbalance")
            score = max(0, 1 - height_diff / 40)
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _check_spine_straightness(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check spine straightness."""
        issues = []
        recommendations = []
        
        # Get key spine points
        nose = self._get_keypoint(keypoints, 0)  # nose
        left_shoulder = self._get_keypoint(keypoints, 5)  # left shoulder
        right_shoulder = self._get_keypoint(keypoints, 6)  # right shoulder
        left_hip = self._get_keypoint(keypoints, 11)  # left hip
        right_hip = self._get_keypoint(keypoints, 12)  # right hip
        
        if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.5, ["Insufficient spine data"], ["Ensure full body is visible"]
        
        # Calculate spine alignment
        shoulder_center = ((left_shoulder['x'] + right_shoulder['x']) / 2,
                          (left_shoulder['y'] + right_shoulder['y']) / 2)
        hip_center = ((left_hip['x'] + right_hip['x']) / 2,
                     (left_hip['y'] + right_hip['y']) / 2)
        
        # Check if spine is straight (vertical)
        spine_angle = self._calculate_angle(shoulder_center, hip_center)
        vertical_deviation = abs(spine_angle - 90)  # 90 degrees is vertical
        
        if vertical_deviation > 15:  # degrees
            issues.append("Spine not straight")
            recommendations.append("Try to maintain a straight back posture")
            score = max(0, 1 - vertical_deviation / 30)
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _analyze_sitting_posture(
        self, 
        keypoints: List[Dict[str, Any]], 
        standards: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze sitting posture."""
        issues = []
        recommendations = []
        scores = []
        
        # Back support analysis
        back_score, back_issues, back_recs = self._check_back_support(keypoints)
        scores.append(back_score * standards['back_support']['weight'])
        issues.extend(back_issues)
        recommendations.extend(back_recs)
        
        # Leg position analysis
        leg_score, leg_issues, leg_recs = self._check_leg_position(keypoints)
        scores.append(leg_score * standards['leg_position']['weight'])
        issues.extend(leg_issues)
        recommendations.extend(leg_recs)
        
        # Head position analysis
        head_score, head_issues, head_recs = self._check_head_position_sitting(keypoints)
        scores.append(head_score * standards['head_position']['weight'])
        issues.extend(head_issues)
        recommendations.extend(head_recs)
        
        overall_score = sum(scores)
        
        return overall_score, issues, recommendations
    
    def _check_back_support(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check back support in sitting posture."""
        issues = []
        recommendations = []
        
        # Get key points for back analysis
        left_shoulder = self._get_keypoint(keypoints, 5)
        right_shoulder = self._get_keypoint(keypoints, 6)
        left_hip = self._get_keypoint(keypoints, 11)
        right_hip = self._get_keypoint(keypoints, 12)
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.5, ["Insufficient back data"], ["Ensure back is visible"]
        
        # Calculate back angle
        shoulder_center = ((left_shoulder['x'] + right_shoulder['x']) / 2,
                          (left_shoulder['y'] + right_shoulder['y']) / 2)
        hip_center = ((left_hip['x'] + right_hip['x']) / 2,
                     (left_hip['y'] + right_hip['y']) / 2)
        
        back_angle = self._calculate_angle(shoulder_center, hip_center)
        
        # Ideal sitting back angle is around 100-110 degrees
        if back_angle < 80:
            issues.append("Excessive forward lean")
            recommendations.append("Sit back in your chair with proper back support")
            score = 0.3
        elif back_angle > 120:
            issues.append("Excessive backward lean")
            recommendations.append("Sit up straight with your back against the chair")
            score = 0.6
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _check_leg_position(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check leg position in sitting posture."""
        issues = []
        recommendations = []
        
        left_knee = self._get_keypoint(keypoints, 13)
        right_knee = self._get_keypoint(keypoints, 14)
        left_ankle = self._get_keypoint(keypoints, 15)
        right_ankle = self._get_keypoint(keypoints, 16)
        
        if not all([left_knee, right_knee, left_ankle, right_ankle]):
            return 0.5, ["Insufficient leg data"], ["Ensure legs are visible"]
        
        # Check if feet are flat on the ground
        hip_center_y = (self._get_keypoint(keypoints, 11)['y'] + 
                       self._get_keypoint(keypoints, 12)['y']) / 2
        ankle_center_y = (left_ankle['y'] + right_ankle['y']) / 2
        
        if ankle_center_y < hip_center_y - 50:  # feet not on ground
            issues.append("Feet not properly positioned")
            recommendations.append("Keep feet flat on the ground")
            score = 0.6
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _check_head_position_sitting(
        self, 
        keypoints: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Check head position in sitting posture."""
        issues = []
        recommendations = []
        
        nose = self._get_keypoint(keypoints, 0)
        left_shoulder = self._get_keypoint(keypoints, 5)
        right_shoulder = self._get_keypoint(keypoints, 6)
        
        if not all([nose, left_shoulder, right_shoulder]):
            return 0.5, ["Insufficient head data"], ["Ensure head is visible"]
        
        # Check if head is forward (forward head posture)
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        head_forward = nose['x'] - shoulder_center_x
        
        if head_forward > 30:  # pixels
            issues.append("Forward head posture")
            recommendations.append("Keep your head aligned over your shoulders")
            score = max(0, 1 - head_forward / 60)
        else:
            score = 1.0
        
        return score, issues, recommendations
    
    def _analyze_walking_posture(
        self, 
        keypoints: List[Dict[str, Any]], 
        standards: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """Analyze walking posture."""
        # This would require temporal analysis of walking gait
        # For now, return a placeholder implementation
        issues = ["Walking analysis requires temporal data"]
        recommendations = ["Record walking sequence for analysis"]
        
        return 0.5, issues, recommendations
    
    def _get_keypoint(self, keypoints: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
        """Get keypoint by index with confidence check."""
        if index < len(keypoints):
            kp = keypoints[index]
            if kp.get('confidence', 0) > 0.5:
                return kp
        return None
    
    def _calculate_angle(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate angle between two points relative to vertical."""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        if dx == 0:
            return 90.0  # Vertical line
        
        angle = np.arctan(dy / dx) * 180 / np.pi
        return abs(angle)


class HealthcareMonitor:
    """Main healthcare monitoring application."""
    
    def __init__(self, model_type: str = 'mediapipe'):
        """
        Initialize healthcare monitor.
        
        Args:
            model_type: Type of pose estimation model to use
        """
        self.detector = PoseDetector(model_type=model_type)
        self.posture_analyzer = PostureAnalyzer()
        
        # Monitoring state
        self.current_session = None
        self.assessment_history = deque(maxlen=100)  # Keep last 100 assessments
        self.is_monitoring = False
        
        # Alert system
        self.alert_threshold = 0.3  # Posture score below this triggers alert
        self.alert_cooldown = 30  # Seconds between alerts
        self.last_alert_time = 0
    
    def start_monitoring_session(
        self, 
        patient_id: str, 
        exercise_type: str = 'general'
    ) -> RehabilitationSession:
        """
        Start a new monitoring session.
        
        Args:
            patient_id: Patient identifier
            exercise_type: Type of exercise/activity
        
        Returns:
            RehabilitationSession object
        """
        session_id = f"{patient_id}_{int(time.time())}"
        
        self.current_session = RehabilitationSession(
            session_id=session_id,
            patient_id=patient_id,
            exercise_type=exercise_type,
            start_time=time.time(),
            end_time=0.0,
            assessments=[],
            progress_score=0.0,
            recommendations=[]
        )
        
        self.is_monitoring = True
        print(f"Started monitoring session for patient {patient_id}")
        
        return self.current_session
    
    def stop_monitoring_session(self) -> RehabilitationSession:
        """
        Stop the current monitoring session.
        
        Returns:
            Completed RehabilitationSession object
        """
        if not self.current_session:
            raise ValueError("No active monitoring session")
        
        self.current_session.end_time = time.time()
        
        # Calculate progress score
        if self.current_session.assessments:
            posture_scores = [assessment.posture_score for assessment in self.current_session.assessments]
            self.current_session.progress_score = np.mean(posture_scores)
            
            # Generate recommendations based on session data
            self.current_session.recommendations = self._generate_recommendations(
                self.current_session.assessments
            )
        
        # Reset monitoring state
        self.is_monitoring = False
        session = self.current_session
        self.current_session = None
        
        print(f"Completed monitoring session for patient {session.patient_id}")
        print(f"Progress score: {session.progress_score:.2f}")
        
        return session
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        posture_type: str = 'standing'
    ) -> Dict[str, Any]:
        """
        Process a single frame for posture monitoring.
        
        Args:
            frame: Input video frame
            posture_type: Type of posture to analyze
        
        Returns:
            Dictionary containing monitoring results
        """
        if not self.is_monitoring:
            return {'status': 'not_monitoring'}
        
        # Detect poses
        results = self.detector.detect_poses(frame, return_visualization=False)
        
        if not results['predictions']:
            return {'status': 'no_pose_detected'}
        
        # Get first person's keypoints
        keypoints = results['predictions'][0].get('keypoints', [])
        
        # Analyze posture
        assessment = self.posture_analyzer.analyze_posture(keypoints, posture_type)
        
        # Add to session
        if self.current_session:
            self.current_session.assessments.append(assessment)
        
        # Add to history
        self.assessment_history.append(assessment)
        
        # Check for alerts
        alert_status = self._check_alerts(assessment)
        
        return {
            'status': 'monitoring',
            'assessment': assessment,
            'alert_status': alert_status,
            'session_stats': self._get_session_stats()
        }
    
    def _check_alerts(self, assessment: PostureAssessment) -> Dict[str, Any]:
        """Check if posture requires immediate attention."""
        current_time = time.time()
        
        # Check if enough time has passed since last alert
        if current_time - self.last_alert_time < self.alert_cooldown:
            return {'alert': False, 'message': 'Alert cooldown active'}
        
        # Check posture score
        if assessment.posture_score < self.alert_threshold:
            self.last_alert_time = current_time
            return {
                'alert': True,
                'message': 'Poor posture detected!',
                'recommendations': assessment.recommendations[:2]  # Top 2 recommendations
            }
        
        return {'alert': False, 'message': 'Posture within acceptable range'}
    
    def _get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        if not self.current_session:
            return {}
        
        duration = time.time() - self.current_session.start_time
        
        if self.current_session.assessments:
            posture_scores = [assessment.posture_score for assessment in self.current_session.assessments]
            avg_posture = np.mean(posture_scores)
            min_posture = np.min(posture_scores)
            
            # Count issues
            all_issues = []
            for assessment in self.current_session.assessments:
                all_issues.extend(assessment.issues)
            
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            most_common_issue = max(issue_counts.items(), key=lambda x: x[1]) if issue_counts else None
        else:
            avg_posture = 0
            min_posture = 0
            most_common_issue = None
        
        return {
            'patient_id': self.current_session.patient_id,
            'exercise_type': self.current_session.exercise_type,
            'duration_minutes': duration / 60,
            'assessments_count': len(self.current_session.assessments),
            'average_posture_score': avg_posture,
            'min_posture_score': min_posture,
            'most_common_issue': most_common_issue
        }
    
    def _generate_recommendations(self, assessments: List[PostureAssessment]) -> List[str]:
        """Generate recommendations based on session assessments."""
        recommendations = []
        
        if not assessments:
            return ["No data available for recommendations"]
        
        # Analyze common issues
        all_issues = []
        for assessment in assessments:
            all_issues.extend(assessment.issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate recommendations based on most common issues
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            if count > len(assessments) * 0.3:  # Issue occurs in >30% of assessments
                if "head" in issue.lower():
                    recommendations.append("Focus on maintaining proper head alignment")
                elif "shoulder" in issue.lower():
                    recommendations.append("Practice shoulder relaxation exercises")
                elif "spine" in issue.lower() or "back" in issue.lower():
                    recommendations.append("Strengthen core muscles and practice good posture")
                elif "hip" in issue.lower():
                    recommendations.append("Check for muscle imbalances and consider physical therapy")
        
        # Add general recommendations
        avg_score = np.mean([assessment.posture_score for assessment in assessments])
        if avg_score < 0.6:
            recommendations.append("Consider regular posture exercises and ergonomic adjustments")
        elif avg_score < 0.8:
            recommendations.append("Continue practicing good posture habits")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_patient_history(self, patient_id: str, days: int = 7) -> List[RehabilitationSession]:
        """Get patient's monitoring history."""
        # In a real implementation, this would query a database
        # For now, return sessions from current monitoring history
        patient_sessions = []
        
        if self.current_session and self.current_session.patient_id == patient_id:
            patient_sessions.append(self.current_session)
        
        return patient_sessions
    
    def export_session_report(self, session: RehabilitationSession, output_path: str):
        """Export session report to file."""
        report = {
            'session_id': session.session_id,
            'patient_id': session.patient_id,
            'exercise_type': session.exercise_type,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'duration_minutes': (session.end_time - session.start_time) / 60,
            'progress_score': session.progress_score,
            'total_assessments': len(session.assessments),
            'recommendations': session.recommendations,
            'assessment_summary': {
                'average_posture_score': np.mean([a.posture_score for a in session.assessments]) if session.assessments else 0,
                'min_posture_score': np.min([a.posture_score for a in session.assessments]) if session.assessments else 0,
                'max_posture_score': np.max([a.posture_score for a in session.assessments]) if session.assessments else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Session report exported to: {output_path}")
