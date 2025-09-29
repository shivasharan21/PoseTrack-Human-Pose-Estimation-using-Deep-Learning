"""
Real-time pose detection demo using webcam.
"""

import cv2
import numpy as np
import sys
import os
import argparse
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import RealtimePoseDetector
from src.applications import FitnessTracker, HealthcareMonitor, RealTimeActivityTracker


def main():
    """Main function for real-time pose detection demo."""
    parser = argparse.ArgumentParser(description='Real-time Pose Detection Demo')
    parser.add_argument('--model', type=str, default='mediapipe', 
                       choices=['mediapipe', 'openpose', 'hrnet'],
                       help='Model type to use')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera ID')
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 480],
                       help='Video resolution (width height)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--application', type=str, default='basic',
                       choices=['basic', 'fitness', 'healthcare', 'activity'],
                       help='Application type')
    parser.add_argument('--exercise', type=str, default='push_up',
                       help='Exercise type for fitness tracking')
    parser.add_argument('--patient_id', type=str, default='demo_patient',
                       help='Patient ID for healthcare monitoring')
    
    args = parser.parse_args()
    
    print("Real-time Pose Detection Demo")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Application: {args.application}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print("Press 'q' to quit")
    print("=" * 40)
    
    # Create detector
    detector = RealtimePoseDetector(
        model_type=args.model,
        camera_id=args.camera,
        resolution=tuple(args.resolution),
        confidence_threshold=args.confidence,
        smoothing=True
    )
    
    # Create application-specific tracker
    if args.application == 'fitness':
        tracker = FitnessTracker(model_type=args.model)
        print(f"Starting {args.exercise} tracking session...")
        tracker.start_exercise_session(args.exercise)
        
        def process_frame(frame):
            result = tracker.process_frame(frame)
            return result
    elif args.application == 'healthcare':
        monitor = HealthcareMonitor(model_type=args.model)
        print(f"Starting healthcare monitoring for patient {args.patient_id}...")
        monitor.start_monitoring_session(args.patient_id)
        
        def process_frame(frame):
            result = monitor.process_frame(frame)
            return result
    elif args.application == 'activity':
        activity_tracker = RealTimeActivityTracker(model_type=args.model)
        print("Starting activity recognition...")
        activity_tracker.start_tracking()
        
        def process_frame(frame):
            result = activity_tracker.process_frame(frame)
            return result
    else:
        def process_frame(frame):
            # Basic pose detection
            results = detector.detect_poses(frame, return_visualization=True)
            return {
                'status': 'tracking',
                'predictions': results.get('predictions', []),
                'visualization': results.get('visualization', frame)
            }
    
    # Start camera
    if not detector.start_camera():
        print("Failed to start camera")
        return
    
    try:
        while True:
            ret, frame = detector.cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, tuple(args.resolution))
            
            # Process frame
            start_time = time.time()
            result = process_frame(frame)
            processing_time = time.time() - start_time
            
            # Display frame
            if 'visualization' in result:
                display_frame = cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame
            
            # Add performance info
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add application-specific info
            if args.application == 'fitness' and 'session_stats' in result:
                stats = result['session_stats']
                cv2.putText(display_frame, f"Reps: {stats.get('total_reps', 0)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Quality: {stats.get('average_quality', 0):.2f}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            elif args.application == 'healthcare' and 'session_stats' in result:
                stats = result['session_stats']
                cv2.putText(display_frame, f"Posture Score: {stats.get('average_posture_score', 0):.2f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if 'alert_status' in result and result['alert_status']['alert']:
                    cv2.putText(display_frame, "POSTURE ALERT!", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif args.application == 'activity' and 'tracking_stats' in result:
                stats = result['tracking_stats']
                if stats.get('current_activity'):
                    cv2.putText(display_frame, f"Activity: {stats['current_activity']}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                if stats.get('current_gesture'):
                    cv2.putText(display_frame, f"Gesture: {stats['current_gesture']}", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Real-time Pose Detection', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        detector.stop_camera()
        cv2.destroyAllWindows()
        
        # Stop application-specific tracking
        if args.application == 'fitness':
            session = tracker.stop_exercise_session()
            print(f"Exercise session completed:")
            print(f"  Total reps: {session.total_reps}")
            print(f"  Average quality: {session.average_quality:.2f}")
            print(f"  Calories burned: {session.calories_burned:.1f}")
        
        elif args.application == 'healthcare':
            session = monitor.stop_monitoring_session()
            print(f"Healthcare monitoring completed:")
            print(f"  Progress score: {session.progress_score:.2f}")
            print(f"  Recommendations: {session.recommendations}")
        
        elif args.application == 'activity':
            activity_tracker.stop_tracking()
            summary = activity_tracker.get_activity_summary(duration_minutes=1)
            print(f"Activity summary:")
            print(f"  Activities detected: {summary['total_activities']}")
            print(f"  Gestures detected: {summary['total_gestures']}")


if __name__ == '__main__':
    main()
