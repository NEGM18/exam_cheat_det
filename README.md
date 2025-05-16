# Exam Cheating Detection System

An AI-powered system for detecting cheating behavior during online exams using computer vision and audio analysis.

## Features

- Real-time video monitoring for suspicious behavior
- Audio monitoring for unusual sounds
- Machine learning model for cheating detection
- Support for recorded video analysis
- Detailed logging and event tracking

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow
- PyAudio
- NumPy
- Matplotlib
- scikit-learn
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/exam_cheat_det.git
cd exam_cheat_det
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train_with_videos.py
```

2. Run the detection system:
```bash
python main.py
```

## Project Structure

- `main.py` - Main application entry point
- `video_monitor.py` - Video monitoring and analysis
- `audio_monitor.py` - Audio monitoring and analysis
- `ml_model.py` - Machine learning model for cheating detection
- `train_with_videos.py` - Model training script
- `detection_utils.py` - Utility functions for detection
- `cheating_videos/` - Directory for cheating behavior videos
- `normal_videos/` - Directory for normal behavior videos

## Model Training

The system uses a CNN-based model trained on video frames to detect cheating behavior. Training data should be organized as follows:

- `cheating_videos/` - Videos showing cheating behavior
- `normal_videos/` - Videos showing normal exam behavior

## License

MIT License 