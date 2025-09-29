"""
Real-time pose detection using webcam or video streams.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, Callable
import threading
from queue import Queue
import argparse

from .detector import PoseDetector
from ..models import MediaPipePose
from ..data.utils import visualize_keypoints, COCO_SKELETON, COCO_KEYPOINT_NAMES


class RealtimePoseDetector:
    """Real-time pose detection for live video streams."""
    
    def __init__(
        self,
        model_type: str = 'mediapipe',
        camera_id: int = 0,
        resolution: tuple = (640, 480),
        fps: int = 30,
        confidence_threshold: float = 0.5,
        smoothing: bool = True,
        smoothing_factor: float = 0.3
    ):
        """
        Initialize real-time pose detector.
        
        Args:
            model_type: Type of model to use
            camera_id: Camera ID for webcam
            resolution: Video resolution (width, height)
            fps: Target FPS
            confidence_threshold: Minimum confidence threshold
            smoothing: Whether to apply smoothing to keypoints
            smoothing_factor: Smoothing factor (0-1)
        """
        self.model_type = model_type
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        
        # Initialize pose detector
        self.detector = PoseDetector(
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            input_size=(resolution[1], resolution[0])  # (height, width)
        )
        
        # Initialize camera
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
        # Performance tracking
        self.frame_times = []
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Smoothing buffers
        self.keypoint_history = []
        self.history_size = 5
        
        # Callbacks
        self.on_keypoints_detected = None
        self.on_frame_processed = None
    
    def set_callback(self, callback_type: str, callback: Callable):
        """
        Set callback functions.
        
        Args:
            callback_type: Type of callback ('keypoints', 'frame')
            callback: Callback function
        """
        if callback_type == 'keypoints':
            self.on_keypoints_detected = callback
        elif callback_type == 'frame':
            self.on_frame_processed = callback
    
    def start_camera(self) -> bool:
        """Start camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def run_webcam(self, display: bool = True, save_video: bool = False, output_path: str = None):
        """
        Run real-time pose detection on webcam.
        
        Args:
            display: Whether to display the video
            save_video: Whether to save the video
            output_path: Output path for saved video
        """
        if not self.start_camera():
            print("Failed to start camera")
            return
        
        # Initialize video writer if saving
        writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, self.resolution
            )
        
        self.is_running = True
        print("Starting real-time pose detection. Press 'q' to quit.")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect poses
                results = self.detector.detect_poses(frame_rgb, return_visualization=True)
                
                # Apply smoothing if enabled
                if self.smoothing and results['predictions']:
                    results['predictions'] = self._apply_smoothing(results['predictions'])
                
                # Call callback if set
                if self.on_keypoints_detected:
                    self.on_keypoints_detected(results['predictions'])
                
                # Update performance stats
                self._update_performance_stats(start_time)
                
                # Display frame
                if display:
                    display_frame = results.get('visualization', frame_rgb)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add FPS counter
                    self._add_fps_counter(display_frame)
                    
                    cv2.imshow('Real-time Pose Detection', display_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Save frame if writer is available
                if writer:
                    if 'visualization' in results:
                        save_frame = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
                    else:
                        save_frame = frame
                    writer.write(save_frame)
                
                # Call frame callback
                if self.on_frame_processed:
                    self.on_frame_processed(frame, results)
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.is_running = False
            self.stop_camera()
            
            if display:
                cv2.destroyAllWindows()
            
            if writer:
                writer.release()
    
    def process_video_stream(
        self, 
        video_path: str, 
        display: bool = True, 
        output_path: Optional[str] = None
    ):
        """
        Process video file in real-time.
        
        Args:
            video_path: Path to video file
            display: Whether to display the video
            output_path: Output path for processed video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if saving
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, self.resolution
            )
        
        self.is_running = True
        print(f"Processing video: {video_path}")
        print(f"Total frames: {frame_count}, FPS: {fps}")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Resize frame to target resolution
                frame = cv2.resize(frame, self.resolution)
                
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect poses
                results = self.detector.detect_poses(frame_rgb, return_visualization=True)
                
                # Apply smoothing if enabled
                if self.smoothing and results['predictions']:
                    results['predictions'] = self._apply_smoothing(results['predictions'])
                
                # Update performance stats
                self._update_performance_stats(start_time)
                
                # Display frame
                if display:
                    display_frame = results.get('visualization', frame_rgb)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add FPS counter
                    self._add_fps_counter(display_frame)
                    
                    cv2.imshow('Video Pose Detection', display_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Save frame if writer is available
                if writer:
                    if 'visualization' in results:
                        save_frame = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
                    else:
                        save_frame = frame
                    writer.write(save_frame)
                
                # Control playback speed
                time.sleep(1.0 / fps)
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.is_running = False
            cap.release()
            
            if display:
                cv2.destroyAllWindows()
            
            if writer:
                writer.release()
    
    def _apply_smoothing(self, predictions: list) -> list:
        """Apply temporal smoothing to keypoints."""
        if not predictions:
            return predictions
        
        # Add current predictions to history
        self.keypoint_history.append(predictions)
        
        # Keep only recent history
        if len(self.keypoint_history) > self.history_size:
            self.keypoint_history.pop(0)
        
        # Apply smoothing
        smoothed_predictions = []
        
        for pred_idx, prediction in enumerate(predictions):
            if 'keypoints' in prediction:
                keypoints = prediction['keypoints']
                smoothed_keypoints = []
                
                for kp_idx, kp in enumerate(keypoints):
                    # Collect historical values for this keypoint
                    historical_values = []
                    for hist_pred in self.keypoint_history:
                        if (kp_idx < len(hist_pred) and 
                            'keypoints' in hist_pred and 
                            kp_idx < len(hist_pred['keypoints'])):
                            historical_values.append(hist_pred['keypoints'][kp_idx])
                    
                    if len(historical_values) > 1:
                        # Calculate smoothed position
                        avg_x = np.mean([kv['x'] for kv in historical_values])
                        avg_y = np.mean([kv['y'] for kv in historical_values])
                        avg_conf = np.mean([kv.get('confidence', 0) for kv in historical_values])
                        
                        # Apply exponential smoothing
                        smoothed_x = self.smoothing_factor * avg_x + (1 - self.smoothing_factor) * kp['x']
                        smoothed_y = self.smoothing_factor * avg_y + (1 - self.smoothing_factor) * kp['y']
                        smoothed_conf = self.smoothing_factor * avg_conf + (1 - self.smoothing_factor) * kp.get('confidence', 0)
                        
                        smoothed_kp = kp.copy()
                        smoothed_kp['x'] = smoothed_x
                        smoothed_kp['y'] = smoothed_y
                        smoothed_kp['confidence'] = smoothed_conf
                        smoothed_keypoints.append(smoothed_kp)
                    else:
                        smoothed_keypoints.append(kp)
                
                smoothed_prediction = prediction.copy()
                smoothed_prediction['keypoints'] = smoothed_keypoints
                smoothed_predictions.append(smoothed_prediction)
            else:
                smoothed_predictions.append(prediction)
        
        return smoothed_predictions
    
    def _update_performance_stats(self, start_time: float):
        """Update performance statistics."""
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > 30:  # Keep last 30 frames
            self.frame_times.pop(0)
        
        # Calculate FPS
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _add_fps_counter(self, frame: np.ndarray):
        """Add FPS counter to frame."""
        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(
            frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.frame_times:
            return {'fps': 0, 'avg_frame_time': 0}
        
        return {
            'fps': self.current_fps,
            'avg_frame_time': np.mean(self.frame_times),
            'min_frame_time': np.min(self.frame_times),
            'max_frame_time': np.max(self.frame_times)
        }
    
    def stop(self):
        """Stop the real-time detection."""
        self.is_running = False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Real-time Pose Detection')
    parser.add_argument('--model', type=str, default='mediapipe', 
                       choices=['mediapipe', 'openpose', 'hrnet'],
                       help='Model type to use')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera ID')
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 480],
                       help='Video resolution (width height)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--video', type=str, default=None,
                       help='Process video file instead of webcam')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--smoothing', action='store_true',
                       help='Enable keypoint smoothing')
    
    args = parser.parse_args()
    
    # Create detector
    detector = RealtimePoseDetector(
        model_type=args.model,
        camera_id=args.camera,
        resolution=tuple(args.resolution),
        confidence_threshold=args.confidence,
        smoothing=args.smoothing
    )
    
    # Run detection
    if args.video:
        detector.process_video_stream(
            args.video, 
            display=not args.no_display,
            output_path=args.output
        )
    else:
        detector.run_webcam(
            display=not args.no_display,
            save_video=args.output is not None,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
