# 🎯 Human Pose Estimation – Deep Learning Implementation

A **comprehensive Human Pose Estimation system** supporting multiple state-of-the-art models such as **MediaPipe, OpenPose, HRNet, and custom CNNs with attention mechanisms**.  
This project covers the full pipeline: **data processing, training, real-time inference, optimization, and domain-specific applications** in fitness, healthcare, surveillance, and HCI.

---

## 📁 Project Structure
Human pose estimation/
├── src/
│ ├── models/ # Pose estimation models (MediaPipe, OpenPose, HRNet)
│ ├── data/ # Data processing and augmentation
│ ├── training/ # Training and evaluation framework
│ ├── inference/ # Real-time detection pipeline
│ ├── applications/ # Domain-specific applications
│ └── optimization/ # Model optimization (quantization, GPU)
├── demos/ # Demo applications
├── configs/ # Configuration files
├── scripts/ # Utility scripts (training, evaluation, preprocessing)
├── notebooks/ # Analysis notebooks
├── requirements.txt # Dependencies
└── README.md # Documentation

yaml
Copy code

---

## 🚀 Features
- **Multiple Models**: MediaPipe, OpenPose, HRNet, Custom CNNs  
- **Data Pipeline**: COCO & MPII support, Albumentations, imgaug  
- **Training Framework**: PyTorch, multiple loss functions, advanced metrics  
- **Inference**: Real-time webcam & video, GPU acceleration, smoothing filters  
- **Applications**:  
  - 🏃 Fitness Tracking – form correction, rep counting, calorie estimation  
  - 🏥 Healthcare – posture monitoring, rehab tracking, fall detection  
  - 🎮 HCI – gesture recognition, VR/AR, gaming  
  - 🎥 Surveillance – activity recognition, crowd monitoring  
- **Optimization**: Quantization, pruning, TensorRT, CUDA  

---

## ⚡ Installation
Clone the repository:
```bash
git clone https://github.com/shivasharan21/PoseTrack-Human-Pose-Estimation-using-Deep-Learning.git
cd PoseTrack-Human-Pose-Estimation-using-Deep-Learning
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
▶️ Running the Applications
1. Real-time Pose Detection (Webcam)
bash
Copy code
python demos/real_time_pose_detection.py
2. Image Pose Detection
bash
Copy code
python demos/image_pose_detection.py --image path/to/image.jpg
3. Video Pose Detection
bash
Copy code
python demos/image_pose_detection.py --video path/to/video.mp4
4. Model Training Demo
bash
Copy code
python demos/model_training_demo.py
5. Model Optimization Demo
bash
Copy code
python demos/optimization_demo.py
⚙️ Using Configs & Scripts
This project uses YAML configuration files to manage models, datasets, and training settings.

Example 1 – Train a Model
bash
Copy code
python scripts/train.py --config configs/hrnet.yaml
Example 2 – Evaluate a Model
bash
Copy code
python scripts/eval.py --config configs/hrnet.yaml --checkpoint checkpoints/hrnet_best.pth
Example 3 – Run Inference on Video
bash
Copy code
python scripts/infer.py --config configs/mediapipe.yaml --video sample_video.mp4
Example 4 – Preprocess Dataset
bash
Copy code
python scripts/preprocess.py --dataset coco --output processed_data/
📊 Performance Highlights
Model Size: 200MB → 12MB (optimized MediaPipe)

Inference Speed: 15ms (MediaPipe) → 120ms (OpenPose)

Accuracy: 65–75% PCK@0.5

GPU Speedup: 2–5× with CUDA/TensorRT

📌 Roadmap
 Add mobile deployment with TensorFlow Lite

 Multi-person tracking in real-time

 Add ONNX model export support

🤝 Contributing
Contributions are welcome!

Fork this repo

Create a new branch (feature-xyz)

Commit your changes

Open a Pull Request

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Developed by Shivasharan Ghalame
📌 GitHub: shivasharan21
