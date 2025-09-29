"""
Image pose detection demo for single images and batch processing.
"""

import cv2
import numpy as np
import sys
import os
import argparse
import glob
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import PoseDetector, VideoProcessor
from src.data.utils import visualize_keypoints


def main():
    """Main function for image pose detection demo."""
    parser = argparse.ArgumentParser(description='Image Pose Detection Demo')
    parser.add_argument('--model', type=str, default='mediapipe', 
                       choices=['mediapipe', 'openpose', 'hrnet'],
                       help='Model type to use')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detection results')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images')
    parser.add_argument('--video', action='store_true',
                       help='Process video file')
    
    args = parser.parse_args()
    
    print("Image Pose Detection Demo")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("=" * 40)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Create detector
    detector = PoseDetector(
        model_type=args.model,
        confidence_threshold=args.confidence
    )
    
    if args.video:
        # Process video
        process_video(detector, args.input, args.output, args.save_results)
    elif args.batch:
        # Process multiple images
        process_image_batch(detector, args.input, args.output, args.save_results)
    else:
        # Process single image
        process_single_image(detector, args.input, args.output, args.save_results)


def process_single_image(detector, input_path, output_dir, save_results):
    """Process a single image."""
    print(f"Processing image: {input_path}")
    
    # Detect poses
    results = detector.detect_poses(input_path, return_visualization=True)
    
    # Display results
    print(f"Detection results:")
    print(f"  Inference time: {results['inference_time']:.3f}s")
    print(f"  Persons detected: {len(results['predictions'])}")
    
    for i, prediction in enumerate(results['predictions']):
        keypoints = prediction.get('keypoints', [])
        print(f"  Person {i+1}: {len(keypoints)} keypoints detected")
        
        # Count visible keypoints
        visible_count = sum(1 for kp in keypoints if kp.get('confidence', 0) > 0.5)
        print(f"    Visible keypoints: {visible_count}/{len(keypoints)}")
    
    # Save results
    if save_results:
        input_name = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{input_name}_result.jpg")
        
        # Save visualization
        if 'visualization' in results:
            vis_image = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, vis_image)
            print(f"Result saved: {output_path}")
        
        # Save detection data
        detector.save_results(results, output_path)
    
    # Display image
    if 'visualization' in results:
        display_image(results['visualization'])


def process_image_batch(detector, input_dir, output_dir, save_results):
    """Process multiple images."""
    # Get image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    total_time = 0
    total_persons = 0
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Detect poses
        results = detector.detect_poses(image_path, return_visualization=True)
        
        total_time += results['inference_time']
        total_persons += len(results['predictions'])
        
        # Save results
        if save_results:
            input_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_result.jpg")
            
            # Save visualization
            if 'visualization' in results:
                vis_image = cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, vis_image)
            
            # Save detection data
            detector.save_results(results, output_path)
    
    # Print summary
    print(f"\nBatch processing completed:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total persons detected: {total_persons}")
    print(f"  Average inference time: {total_time/len(image_files):.3f}s")
    print(f"  Average persons per image: {total_persons/len(image_files):.1f}")


def process_video(detector, video_path, output_dir, save_results):
    """Process video file."""
    print(f"Processing video: {video_path}")
    
    # Create video processor
    processor = VideoProcessor(detector, output_dir)
    
    # Process video
    results = processor.process_video(
        video_path,
        output_video=os.path.join(output_dir, 'processed_video.mp4') if save_results else None,
        save_keypoints=save_results,
        save_visualizations=save_results
    )
    
    # Print results
    stats = results['stats']
    print(f"Video processing completed:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Processed frames: {stats['processed_frames']}")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Total time: {stats['total_time']:.1f}s")


def display_image(image):
    """Display image with pose detection results."""
    # Convert RGB to BGR for OpenCV display
    display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize if too large
    height, width = display_image.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200/width, 800/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_image = cv2.resize(display_image, (new_width, new_height))
    
    cv2.imshow('Pose Detection Result', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def analyze_pose_data(results):
    """Analyze pose detection results."""
    if not results['predictions']:
        print("No poses detected")
        return
    
    print("\nPose Analysis:")
    
    for i, prediction in enumerate(results['predictions']):
        keypoints = prediction.get('keypoints', [])
        
        if not keypoints:
            continue
        
        print(f"\nPerson {i+1}:")
        
        # Calculate pose statistics
        visible_keypoints = [kp for kp in keypoints if kp.get('confidence', 0) > 0.5]
        
        if len(visible_keypoints) >= 2:
            # Calculate pose center
            center_x = np.mean([kp['x'] for kp in visible_keypoints])
            center_y = np.mean([kp['y'] for kp in visible_keypoints])
            print(f"  Pose center: ({center_x:.1f}, {center_y:.1f})")
            
            # Calculate pose bounds
            x_coords = [kp['x'] for kp in visible_keypoints]
            y_coords = [kp['y'] for kp in visible_keypoints]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            print(f"  Pose dimensions: {width:.1f} x {height:.1f}")
            
            # Calculate average confidence
            avg_confidence = np.mean([kp.get('confidence', 0) for kp in visible_keypoints])
            print(f"  Average confidence: {avg_confidence:.3f}")
        
        # Keypoint analysis
        print(f"  Keypoint visibility:")
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        for j, (kp, name) in enumerate(zip(keypoints, keypoint_names)):
            if j < len(keypoint_names):
                confidence = kp.get('confidence', 0)
                status = "✓" if confidence > 0.5 else "✗"
                print(f"    {name}: {status} ({confidence:.2f})")


if __name__ == '__main__':
    main()
