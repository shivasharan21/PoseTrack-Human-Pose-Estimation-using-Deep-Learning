"""
Pose detection module for single images and batch processing.
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import time
from pathlib import Path

from ..models import MediaPipePose, OpenPoseModel, HRNetModel
from ..data.utils import visualize_keypoints, COCO_SKELETON, COCO_KEYPOINT_NAMES


class PoseDetector:
    """Main pose detection class for single images and batch processing."""
    
    def __init__(
        self,
        model_type: str = 'mediapipe',
        model_path: Optional[str] = None,
        device: str = 'cpu',
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize pose detector.
        
        Args:
            model_type: Type of model ('mediapipe', 'openpose', 'hrnet')
            model_path: Path to model weights (for custom models)
            device: Device to run inference on ('cpu', 'cuda')
            confidence_threshold: Minimum confidence threshold
            input_size: Input image size for model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Initialize model
        self.model = self._load_model()
        
        # Performance tracking
        self.inference_times = []
    
    def _load_model(self):
        """Load the specified model."""
        if self.model_type.lower() == 'mediapipe':
            return MediaPipePose(
                num_keypoints=33,
                input_size=self.input_size,
                static_image_mode=True
            )
        elif self.model_type.lower() == 'openpose':
            model = OpenPoseModel(
                num_keypoints=19,
                input_size=self.input_size
            )
            if self.model_path and Path(self.model_path).exists():
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            return model
        elif self.model_type.lower() == 'hrnet':
            model = HRNetModel(
                num_keypoints=17,
                input_size=self.input_size
            )
            if self.model_path and Path(self.model_path).exists():
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            return model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def detect_poses(
        self, 
        image: Union[str, np.ndarray], 
        return_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Detect poses in a single image.
        
        Args:
            image: Input image (path or numpy array)
            return_visualization: Whether to return visualization
        
        Returns:
            Dictionary containing detection results
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]
        
        # Record inference time
        start_time = time.time()
        
        # Preprocess image
        if self.model_type.lower() == 'mediapipe':
            predictions = self.model.predict_keypoints(image, self.confidence_threshold)
        else:
            # For PyTorch models
            input_tensor = self.model.preprocess_input(image)
            
            with torch.no_grad():
                if self.device == 'cuda' and torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    self.model = self.model.cuda()
                
                output = self.model(input_tensor)
                predictions = self.model.predict_keypoints(output, self.confidence_threshold)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Prepare results
        results = {
            'predictions': predictions,
            'inference_time': inference_time,
            'image_size': original_size,
            'model_type': self.model_type
        }
        
        # Add visualization if requested
        if return_visualization:
            vis_image = self._create_visualization(image, predictions)
            results['visualization'] = vis_image
        
        return results
    
    def detect_batch(
        self, 
        images: List[Union[str, np.ndarray]], 
        return_visualizations: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect poses in a batch of images.
        
        Args:
            images: List of input images (paths or numpy arrays)
            return_visualizations: Whether to return visualizations
        
        Returns:
            List of detection results
        """
        results = []
        
        for image in images:
            result = self.detect_poses(image, return_visualizations)
            results.append(result)
        
        return results
    
    def _create_visualization(
        self, 
        image: np.ndarray, 
        predictions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create visualization of pose detection results."""
        if self.model_type.lower() == 'mediapipe':
            return self.model.visualize_results(image, predictions)
        else:
            # For other models, use generic visualization
            vis_image = image.copy()
            
            for prediction in predictions:
                keypoints = prediction.get('keypoints', [])
                
                # Draw skeleton connections
                if self.model_type.lower() == 'openpose':
                    skeleton = self.model.get_skeleton_connections()
                else:  # HRNet or other models
                    skeleton = COCO_SKELETON
                
                # Draw connections
                for connection in skeleton:
                    if len(connection) == 2:
                        start_idx, end_idx = connection[0], connection[1]
                        
                        if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                            start_kp = keypoints[start_idx]
                            end_kp = keypoints[end_idx]
                            
                            if (start_kp.get('confidence', 0) > 0.5 and 
                                end_kp.get('confidence', 0) > 0.5):
                                
                                start_point = (int(start_kp['x']), int(start_kp['y']))
                                end_point = (int(end_kp['x']), int(end_kp['y']))
                                
                                cv2.line(vis_image, start_point, end_point, (0, 255, 0), 2)
                
                # Draw keypoints
                for i, kp in enumerate(keypoints):
                    if kp.get('confidence', 0) > 0.5:
                        point = (int(kp['x']), int(kp['y']))
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
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.inference_times = []
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        output_path: str,
        save_visualization: bool = True
    ):
        """
        Save detection results to file.
        
        Args:
            results: Detection results
            output_path: Output file path
            save_visualization: Whether to save visualization
        """
        import json
        
        # Prepare data for saving
        save_data = {
            'model_type': results['model_type'],
            'inference_time': results['inference_time'],
            'image_size': results['image_size'],
            'predictions': []
        }
        
        # Convert predictions to serializable format
        for prediction in results['predictions']:
            pred_data = {}
            for key, value in prediction.items():
                if isinstance(value, np.ndarray):
                    pred_data[key] = value.tolist()
                else:
                    pred_data[key] = value
            save_data['predictions'].append(pred_data)
        
        # Save JSON data
        json_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save visualization if requested and available
        if save_visualization and 'visualization' in results:
            vis_path = output_path.replace('.jpg', '_vis.jpg').replace('.png', '_vis.png')
            vis_image = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_path, vis_image)
    
    def process_video_frames(
        self, 
        video_path: str, 
        output_dir: str,
        frame_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Process video frames for pose detection.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            frame_interval: Process every Nth frame
        
        Returns:
            Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        processed_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect poses
                result = self.detect_poses(frame_rgb, return_visualization=True)
                
                # Save results
                frame_filename = f"frame_{frame_count:06d}"
                self.save_results(result, f"{output_dir}/{frame_filename}.jpg")
                
                results.append(result)
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        return {
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'frame_interval': frame_interval,
            'avg_inference_time': np.mean([r['inference_time'] for r in results]),
            'results': results
        }
