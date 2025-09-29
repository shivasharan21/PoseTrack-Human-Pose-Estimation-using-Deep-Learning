"""
Simple image pose detection demo using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os


def main():
    """Simple image pose detection demo."""
    parser = argparse.ArgumentParser(description='Simple Image Pose Detection Demo')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='output_result.jpg', help='Output image path')
    
    args = parser.parse_args()
    
    print("Simple Image Pose Detection Demo")
    print("=" * 40)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Create pose detection object
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # Read input image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image '{args.input}'")
        return
    
    print(f"Processing image: {args.input}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = pose.process(rgb_image)
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Count visible landmarks
        landmarks = results.pose_landmarks.landmark
        visible_count = sum(1 for lm in landmarks if lm.visibility > 0.5)
        
        print(f"Pose detected!")
        print(f"Visible landmarks: {visible_count}/{len(landmarks)}")
        
        # Add info text to image
        cv2.putText(image, f"Pose detected: {visible_count}/{len(landmarks)} landmarks", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("No pose detected in the image")
        cv2.putText(image, "No pose detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save result
    cv2.imwrite(args.output, image)
    print(f"Result saved to: {args.output}")
    
    # Display result
    cv2.imshow('Pose Detection Result', image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
