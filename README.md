# Human Pose Estimation using Deep Learning

A comprehensive computer vision system for detecting and estimating human body keypoints from images and videos using Convolutional Neural Networks (CNNs) and transfer learning.

## Features

- **Multiple Model Support**: MediaPipe, OpenPose-style models, and custom CNN architectures
- **Real-time Processing**: Optimized pipeline for live video stream pose detection
- **Data Augmentation**: Advanced preprocessing pipeline for pose datasets (COCO, MPII)
- **Domain Applications**: Fitness tracking, healthcare monitoring, activity recognition
- **Model Optimization**: Quantization, GPU acceleration, and inference optimization
- **Comprehensive Evaluation**: Benchmark testing on standard datasets

## Project Structure

```
├── src/
│   ├── models/          # Pose estimation model implementations
│   ├── data/            # Data processing and augmentation
│   ├── training/        # Training scripts and utilities
│   ├── inference/       # Real-time inference pipeline
│   └── applications/    # Domain-specific applications
├── data/                # Dataset storage
├── weights/             # Pre-trained model weights
├── notebooks/           # Jupyter notebooks for analysis
├── demos/               # Demo applications
└── tests/               # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shivasharan21/PoseTrack-Human-Pose-Estimation-using-Deep-Learning.git
cd PoseTrack-Human-Pose-Estimation-using-Deep-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained weights (optional):
```bash
python scripts/download_weights.py
```

## Quick Start

### Real-time Pose Detection
```python
from src.inference.realtime_detector import PoseDetector

detector = PoseDetector(model_type='mediapipe')
detector.run_webcam()
```

### Process Single Image
```python
from src.inference.detector import detect_poses

keypoints = detect_poses('path/to/image.jpg', model_type='mediapipe')
```

### Training Custom Model
```python
from src.training.trainer import PoseTrainer

trainer = PoseTrainer(config='configs/hrnet_config.yaml')
trainer.train()
```

## Applications

### Fitness Tracking
- Exercise form correction
- Repetition counting
- Movement analysis

### Healthcare Monitoring
- Posture assessment
- Rehabilitation tracking
- Fall detection

### Activity Recognition
- Sports analysis
- Gesture recognition
- Human-computer interaction

## Model Performance

| Model | COCO AP | Inference Time (ms) | Model Size (MB) |
|-------|---------|-------------------|-----------------|
| MediaPipe | 0.65 | 15 | 12 |
| HRNet-W32 | 0.75 | 45 | 28 |
| OpenPose | 0.61 | 120 | 200 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
