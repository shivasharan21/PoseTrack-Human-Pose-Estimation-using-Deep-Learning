"""
Domain-specific applications for pose estimation.

This module contains specialized applications including:
- Fitness tracking and exercise form correction
- Healthcare monitoring and rehabilitation
- Activity recognition and gesture detection
- Sports analysis and performance tracking
"""

from .fitness_tracker import FitnessTracker, ExerciseAnalyzer
from .healthcare_monitor import HealthcareMonitor, PostureAnalyzer
from .activity_recognizer import ActivityRecognizer, GestureDetector
from .sports_analyzer import SportsAnalyzer, PerformanceTracker

__all__ = [
    'FitnessTracker', 'ExerciseAnalyzer',
    'HealthcareMonitor', 'PostureAnalyzer', 
    'ActivityRecognizer', 'GestureDetector',
    'SportsAnalyzer', 'PerformanceTracker'
]
