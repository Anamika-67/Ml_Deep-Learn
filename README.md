# Ml_Deep-Learn
Sherlock Files is a deep learning project that reconstructs the correct chronological order of shuffled video frames. Using CNN for feature extraction and LSTM for temporal modeling, the system learns motion patterns and predicts frame rankings to restore the true sequence of events in videos.

Sherlock Files
Reconstructing the True Timeline of Motion using Deep Learning

Sherlock Files is an AI-powered system designed to reconstruct the correct chronological order of shuffled video frames using computer vision and deep learning. The project analyzes motion dynamics and visual patterns in video frames to restore the original sequence of events.

The system mimics the reasoning of a forensic investigator by identifying temporal relationships between frames and learning how objects move according to physical laws.

This project was developed as part of the MLWARE competition at TECHNEX'26, where participants build machine learning models capable of solving complex temporal reconstruction problems.

Problem Statement

In many real-world scenarios, video data may become corrupted or fragmented, causing frames to lose their correct chronological order.

The objective of this project is to build a deep learning model that predicts the correct ordering of frames in a shuffled video sequence.

Each video contains motion-based interactions such as:

Objects falling

Collisions

Sliding motion

Physical interactions

By analyzing spatial features and temporal relationships between frames, the model reconstructs the correct timeline.

Key Features

Automated video frame extraction

Deep learning based feature extraction using CNN

Temporal sequence learning using LSTM

Frame ranking prediction model

Interactive web interface for video upload and prediction

CSV submission generator for evaluation

Tech Stack
Programming Language

Python

Machine Learning

PyTorch

Computer Vision

OpenCV

Data Processing

NumPy

Pandas

Web Interface

HTML

CSS

JavaScript

FastAPI

System Architecture
Video Input
      ↓
Frame Extraction (OpenCV)
      ↓
Image Preprocessing
      ↓
CNN Feature Extraction
      ↓
Temporal Modeling (LSTM)
      ↓
Frame Ranking Prediction
      ↓
Frame Reordering
      ↓
CSV Output
Project Structure
mlware_sherlock/
│
├── data/
├── frames/
├── models/
│   ├── cnn_encoder.py
│   ├── temporal_model.py
│
├── utils/
│   ├── extract_frames.py
│   ├── preprocessing.py
│   ├── dataset.py
│
├── train.py
├── inference.py
├── generate_submission.py
│
├── ui/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── requirements.txt
Model Architecture

The system consists of two main components:

CNN Feature Extractor

A pretrained CNN model (ResNet50) is used to extract spatial features from each frame.

Temporal Sequence Model

An LSTM network analyzes the sequence of frame features to learn motion patterns and predict temporal relationships.

The model outputs a ranking score for each frame, which is then sorted to reconstruct the correct order.

Evaluation Metric

The model performance is evaluated using Kendall Rank Correlation (Kendall's Tau).

This metric measures how closely the predicted frame order matches the true chronological sequence.

Score Range:

1 → Perfect ordering

0 → Random ordering

-1 → Completely incorrect ordering

How to Run the Project
1. Clone the repository
git clone https://github.com/yourusername/mlware-sherlock-files.git
2. Install dependencies
pip install -r requirements.txt
3. Extract frames
python utils/extract_frames.py
4. Train the model
python train.py
5. Run inference
python inference.py
6. Generate submission file
python generate_submission.py
Future Improvements

Transformer based temporal modeling

Optical flow based motion analysis

Physics-aware motion prediction

Attention visualization for interpretability

Real-time video reconstruction interface

Use Cases

Video forensics

Surveillance analysis

Motion understanding in robotics

Physics-aware AI systems

Temporal reasoning in computer vision
