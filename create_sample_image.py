"""
Create a sample image for testing pose detection.
"""

import cv2
import numpy as np

def create_sample_image():
    """Create a simple sample image with a person silhouette."""
    # Create a white background
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a simple person silhouette
    # Head
    cv2.circle(image, (320, 80), 30, (0, 0, 0), -1)
    
    # Body
    cv2.rectangle(image, (300, 110), (340, 250), (0, 0, 0), -1)
    
    # Arms
    cv2.rectangle(image, (280, 130), (300, 200), (0, 0, 0), -1)  # Left arm
    cv2.rectangle(image, (340, 130), (360, 200), (0, 0, 0), -1)  # Right arm
    
    # Legs
    cv2.rectangle(image, (310, 250), (330, 400), (0, 0, 0), -1)  # Left leg
    cv2.rectangle(image, (330, 250), (350, 400), (0, 0, 0), -1)  # Right leg
    
    # Add some text
    cv2.putText(image, "Sample Person for Pose Detection", (50, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return image

if __name__ == '__main__':
    sample_image = create_sample_image()
    cv2.imwrite('sample_person.jpg', sample_image)
    print("Sample image created: sample_person.jpg")
    
    # Display the image
    cv2.imshow('Sample Person', sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
