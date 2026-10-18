# Cozmo ML Autonomous Vision & Driving System

An end-to-end Machine Learning and Computer Vision suite for the Anki/Digital Dream Labs Cozmo robot using PyTorch, OpenCV, Dlib, and PyCozmo.

This repository implements multiple autonomous control paradigms, ranging from closed-form visual regression and convolutional neural network (CNN) steering to multi-lane navigation with real-time OCR digit recognition and vision-based object distance estimation.

---

## Key Modules

| Module | Architecture / Method | Description |
| :--- | :--- | :--- |
| **`line_follower/`** | Multi-Layer Perceptron (PyTorch) + Linear Regression | Real-time line tracking using visual feature extraction, spatial error history, and neural motor prediction. |
| **`cnn_lf/`** | End-to-End ConvNet (PyTorch) | Image-to-steering Convolutional Neural Network trained directly on raw grayscale camera feed ROIs. |
| **`Multi_lane/`** | Object Detection (Dlib SVM) + OCR Digit Classification | Multi-lane path planning featuring real-time visual digit classification (MNIST-style safetensors MLP) and object detection. |
| **`vision_cube/`** | Visual Geometry Regression | Distance estimation derived from contour pixel bounding box width using calibrated inverse regression. |

---

## Repository Structure

```
cozmo_ml_project/
├── line_follower/         # MLP & Linear Regression line-following pipeline
│   ├── auto_drive.py      # Live neural line-following control loop
│   ├── collect_data.py    # Manual/teleop data collection script
│   ├── linear_reg.py      # Direct closed-form regression line tracker
│   └── train_model.py     # PyTorch training script for AdvancedDrivingNet
├── cnn_lf/                # Convolutional Neural Network line follower
│   ├── auto_drive.py      # End-to-end CNN inference control loop
│   ├── collect_data.py    # Frame & telemetry recording script
│   └── train_model.py     # ConvNet training pipeline with image augmentation
├── Multi_lane/            # Multi-lane navigation & OCR digit classifier
│   ├── lane_detection.py  # Multi-lane detection, obstacle avoidance, & OCR
│   ├── detector.svm       # Dlib object detector model
│   └── model.safetensors  # Safetensors weight file for digit classification MLP
├── vision_cube/           # Distance tracking module
│   ├── inference.py       # Live distance telemetry stream
│   ├── loop.py            # Closed-loop distance maintaining controller
│   └── train_model.py     # Inverse regression calibration utility
├── .env.example           # Template for local environment variables
├── .gitignore              # Git ignore rules for virtualenvs & local artifacts
└── requirements.txt       # Project dependencies
```

---

## Prerequisites & Installation

### Hardware Requirements
- **Anki Cozmo Robot** (with charged battery)
- **Computer with Wi-Fi** (to connect directly to Cozmo's Access Point)

### Software Requirements
- Python 3.9+
- OpenCV 4.7+
- PyTorch 2.0+

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cozmo_ml_project.git
   cd cozmo_ml_project
   ```

2. **Create and activate a Python virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment settings (optional)**:
   ```bash
   cp .env.example .env
   ```

---

## Usage Guide

### 1. MLP Line Follower (`line_follower/`)
- **Collect Driving Data**:
  ```bash
  python -m line_follower.collect_data
  ```
  *Use Arrow keys to drive Cozmo manually while dataset is saved to `driving_data.csv`.*

- **Train Neural Model**:
  ```bash
  python -m line_follower.train_model
  ```

- **Run Autonomous Driving**:
  ```bash
  python -m line_follower.auto_drive
  ```

---

### 2. CNN Line Follower (`cnn_lf/`)
- **Record Dataset**:
  ```bash
  python -m cnn_lf.collect_data
  ```

- **Train ConvNet**:
  ```bash
  python -m cnn_lf.train_model
  ```

- **Run Autonomous Driving**:
  ```bash
  python -m cnn_lf.auto_drive
  ```

---

### 3. Multi-Lane Navigation & OCR (`Multi_lane/`)
- **Run Lane & Digit Classifier Navigation**:
  ```bash
  python -m Multi_lane.lane_detection
  ```

---

### 4. Vision Cube Distance Tracking (`vision_cube/`)
- **Run Live Telemetry**:
  ```bash
  python -m vision_cube.inference
  ```

- **Run Closed-Loop Distance Controller**:
  ```bash
  python -m vision_cube.loop
  ```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
