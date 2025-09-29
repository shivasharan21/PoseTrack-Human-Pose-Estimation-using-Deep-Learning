"""
Test script to verify installation and identify issues.
"""

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
        print(f"  MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
        print(f"  Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_mediapipe():
    """Test MediaPipe pose detection."""
    print("\nTesting MediaPipe pose detection...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # Create a simple test image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Test pose detection
        results = pose.process(test_image)
        print("✓ MediaPipe pose detection working")
        
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def test_camera():
    """Test camera access."""
    print("\nTesting camera access...")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✓ Camera accessible")
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera can capture frames: {frame.shape}")
            else:
                print("✗ Camera cannot capture frames")
            cap.release()
            return True
        else:
            print("✗ Camera not accessible")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("HUMAN POSE ESTIMATION - INSTALLATION TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test MediaPipe
        mediapipe_ok = test_mediapipe()
        
        # Test camera
        camera_ok = test_camera()
        
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        if imports_ok and mediapipe_ok:
            print("✓ All core functionality working!")
            print("You can now run the pose detection demos.")
            
            if camera_ok:
                print("✓ Camera is available - real-time demo should work")
            else:
                print("⚠ Camera not available - only image demo will work")
                
        else:
            print("✗ Some issues found. Please check the error messages above.")
    
    else:
        print("\n✗ Import issues found. Please install missing packages:")
        print("pip install opencv-python mediapipe numpy matplotlib")

if __name__ == '__main__':
    main()

