"""
Video processing utilities for pose estimation.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
import os
from pathlib import Path
import json

from .detector import PoseDetector


class VideoProcessor:
    """Video processing class for pose estimation on video files."""
    
    def __init__(
        self,
        detector: PoseDetector,
        output_dir: str = "output",
        frame_skip: int = 1,
        batch_size: int = 1
    ):
        """
        Initialize video processor.
        
        Args:
            detector: Pose detector instance
            output_dir: Output directory for results
            frame_skip: Process every Nth frame
            batch_size: Batch size for processing
        """
        self.detector = detector
        self.output_dir = output_dir
        self.frame_skip = frame_skip
        self.batch_size = batch_size
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_time': 0,
            'avg_fps': 0,
            'errors': []
        }
    
    def process_video(
        self, 
        video_path: str, 
        output_video: Optional[str] = None,
        save_keypoints: bool = True,
        save_visualizations: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a video file for pose estimation.
        
        Args:
            video_path: Path to input video
            output_video: Path to output video (optional)
            save_keypoints: Whether to save keypoint data
            save_visualizations: Whether to save visualization frames
            progress_callback: Callback function for progress updates
        
        Returns:
            Processing results and statistics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize output video writer if requested
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Initialize processing
        self.stats['total_frames'] = frame_count
        start_time = time.time()
        
        frame_idx = 0
        processed_idx = 0
        results = []
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {frame_count}, FPS: {fps}, Resolution: {width}x{height}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue
                
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect poses
                    result = self.detector.detect_poses(frame_rgb, return_visualization=True)
                    
                    # Add frame information
                    result['frame_idx'] = frame_idx
                    result['timestamp'] = frame_idx / fps
                    
                    results.append(result)
                    
                    # Save keypoints if requested
                    if save_keypoints:
                        self._save_frame_keypoints(result, processed_idx)
                    
                    # Save visualization if requested
                    if save_visualizations:
                        self._save_frame_visualization(result, processed_idx)
                    
                    # Write to output video if writer is available
                    if writer and 'visualization' in result:
                        vis_frame = cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR)
                        writer.write(vis_frame)
                    
                    processed_idx += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress = (frame_idx + 1) / frame_count
                        progress_callback(progress, frame_idx + 1, frame_count)
                    
                    # Print progress
                    if processed_idx % 100 == 0:
                        print(f"Processed {processed_idx} frames ({frame_idx + 1}/{frame_count})")
                
                except Exception as e:
                    error_msg = f"Error processing frame {frame_idx}: {str(e)}"
                    print(error_msg)
                    self.stats['errors'].append(error_msg)
                
                frame_idx += 1
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            
            # Update statistics
            self.stats['processed_frames'] = processed_idx
            self.stats['total_time'] = time.time() - start_time
            self.stats['avg_fps'] = processed_idx / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
        
        # Save summary
        self._save_processing_summary(results, video_path)
        
        return {
            'results': results,
            'stats': self.stats,
            'video_info': {
                'path': video_path,
                'fps': fps,
                'frame_count': frame_count,
                'resolution': (width, height)
            }
        }
    
    def process_video_batch(
        self, 
        video_paths: List[str], 
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple video files in batch.
        
        Args:
            video_paths: List of video file paths
            progress_callback: Callback function for progress updates
        
        Returns:
            Batch processing results
        """
        batch_results = []
        total_videos = len(video_paths)
        
        print(f"Processing {total_videos} videos in batch")
        
        for i, video_path in enumerate(video_paths):
            print(f"\nProcessing video {i + 1}/{total_videos}: {video_path}")
            
            try:
                # Create video-specific output directory
                video_name = Path(video_path).stem
                video_output_dir = os.path.join(self.output_dir, video_name)
                
                # Process video
                result = self.process_video(
                    video_path,
                    output_video=os.path.join(video_output_dir, f"{video_name}_processed.mp4"),
                    progress_callback=lambda p, f, t: progress_callback(
                        (i + p) / total_videos, i * 1000 + f, total_videos * 1000
                    ) if progress_callback else None
                )
                
                batch_results.append(result)
                
            except Exception as e:
                error_msg = f"Error processing video {video_path}: {str(e)}"
                print(error_msg)
                self.stats['errors'].append(error_msg)
        
        return {
            'batch_results': batch_results,
            'total_videos': total_videos,
            'successful_videos': len(batch_results),
            'failed_videos': total_videos - len(batch_results)
        }
    
    def extract_keyframes(
        self, 
        video_path: str, 
        threshold: float = 0.3,
        min_interval: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Extract keyframes based on pose changes.
        
        Args:
            video_path: Path to video file
            threshold: Threshold for pose change detection
            min_interval: Minimum interval between keyframes (frames)
        
        Returns:
            List of keyframe information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        keyframes = []
        last_pose = None
        last_keyframe_frame = 0
        
        frame_idx = 0
        
        print("Extracting keyframes...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_skip == 0:
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect poses
                    result = self.detector.detect_poses(frame_rgb, return_visualization=False)
                    
                    # Check for pose changes
                    if result['predictions'] and len(result['predictions']) > 0:
                        current_pose = result['predictions'][0].get('keypoints', [])
                        
                        if last_pose is not None:
                            # Calculate pose change
                            pose_change = self._calculate_pose_change(last_pose, current_pose)
                            
                            # Check if enough time has passed and pose changed significantly
                            if (frame_idx - last_keyframe_frame >= min_interval and 
                                pose_change > threshold):
                                
                                keyframes.append({
                                    'frame_idx': frame_idx,
                                    'timestamp': frame_idx / fps,
                                    'pose_change': pose_change,
                                    'keypoints': current_pose
                                })
                                
                                last_keyframe_frame = frame_idx
                        
                        last_pose = current_pose
                
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"Extracted {len(keyframes)} keyframes")
        return keyframes
    
    def _save_frame_keypoints(self, result: Dict[str, Any], frame_idx: int):
        """Save keypoints for a single frame."""
        keypoints_data = {
            'frame_idx': result['frame_idx'],
            'timestamp': result['timestamp'],
            'inference_time': result['inference_time'],
            'predictions': result['predictions']
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for prediction in keypoints_data['predictions']:
            if 'keypoints' in prediction:
                for kp in prediction['keypoints']:
                    for key, value in kp.items():
                        if isinstance(value, np.ndarray):
                            kp[key] = value.tolist()
        
        # Save to file
        output_path = os.path.join(
            self.output_dir, 
            f"keypoints_{frame_idx:06d}.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(keypoints_data, f, indent=2)
    
    def _save_frame_visualization(self, result: Dict[str, Any], frame_idx: int):
        """Save visualization for a single frame."""
        if 'visualization' in result:
            vis_image = cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR)
            output_path = os.path.join(
                self.output_dir, 
                f"frame_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(output_path, vis_image)
    
    def _save_processing_summary(self, results: List[Dict[str, Any]], video_path: str):
        """Save processing summary."""
        summary = {
            'video_path': video_path,
            'processing_stats': self.stats,
            'total_frames_processed': len(results),
            'avg_inference_time': np.mean([r['inference_time'] for r in results]),
            'model_type': self.detector.model_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _calculate_pose_change(
        self, 
        pose1: List[Dict[str, Any]], 
        pose2: List[Dict[str, Any]]
    ) -> float:
        """Calculate the change between two poses."""
        if len(pose1) != len(pose2):
            return 1.0  # Maximum change if different number of keypoints
        
        total_change = 0.0
        valid_keypoints = 0
        
        for kp1, kp2 in zip(pose1, pose2):
            if (kp1.get('confidence', 0) > 0.5 and 
                kp2.get('confidence', 0) > 0.5):
                
                # Calculate Euclidean distance
                dx = kp1['x'] - kp2['x']
                dy = kp1['y'] - kp2['y']
                distance = np.sqrt(dx**2 + dy**2)
                
                total_change += distance
                valid_keypoints += 1
        
        if valid_keypoints == 0:
            return 0.0
        
        # Normalize by number of valid keypoints and image size
        avg_change = total_change / valid_keypoints
        normalized_change = avg_change / 100.0  # Normalize by approximate image size
        
        return min(normalized_change, 1.0)  # Cap at 1.0
