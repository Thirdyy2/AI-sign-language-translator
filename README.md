# AI Sign Language Translator 🤟

Real-time American Sign Language (ASL) translator using Computer Vision and Deep Learning. Recognizes both static and motion-based gestures with dual CNN-LSTM architecture.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

- **Dual Model Architecture**
  - MLP for static gesture recognition (A-Z)
  - LSTM for motion-based gestures (J, Z)
  
- **Real-time Detection**
  - Live webcam feed processing
  - 20ms inference per frame
  - MediaPipe 21-landmark hand tracking

- **Smart Preprocessing**
  - Wrist-centered normalization
  - Scale-invariant feature extraction
  - Temporal buffering for sequences

- **Professional UI**
  - Tkinter-based interface
  - Model switching (CNN/LSTM)
  - Start/Stop prediction toggle
  - Text output with Delete/Clear

- **Fully Offline**
  - No internet required
  - Locally trained models
  - Privacy-focused design

## 📸 Demo

[Add screenshots or GIF here]

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Webcam
- Windows/Linux/MacOS

### Setup

1. Clone the repository
```bash
git clone https://github.com/reyanshlakra2011-coder/AI-sign-language-translator.git
cd AI-sign-language-translator
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python asl_gui.py
```

## 📊 Technical Details

### Static Gesture Recognition
- **Architecture**: Multi-layer Perceptron (128→64 neurons)
- **Input**: 63 features (21 landmarks × 3 coordinates)
- **Activation**: ReLU + Softmax
- **Dropout**: 0.3

### Motion Gesture Recognition
- **Architecture**: LSTM (128→64 cells)
- **Input**: 30 frames × 63 features
- **Sequence Length**: 30 timesteps
- **Handles**: J, Z and other motion-based signs

### Preprocessing Pipeline
1. MediaPipe hand landmark detection
2. Coordinate extraction (21 points)
3. Wrist-centered normalization
4. Scale normalization
5. Temporal buffering (for LSTM)

## 🎯 Usage

### Capturing Training Data

**For static gestures (A-Z):**
```bash
python capture_landmarks.py
```

**For motion gestures (J, Z):**
```bash
python capture_sequence.py
```

### Training Models

**Train MLP (static):**
```bash
python prepare_data.py
python train_mlp.py
```

**Train LSTM (motion):**
```bash
python prepare_seq_dataset.py
python train_lstm.py
```

### Running the GUI
```bash
python asl_gui.py
```

**Controls:**
- **Start/Stop Prediction**: Toggle model inference
- **Predict Sequence**: Switch to motion-only mode
- **Delete**: Remove last character
- **Clear**: Clear all text

## 📁 Project Structure
```
AI-sign-language-translator/
├── asl_gui.py                    # Main application
├── capture_landmarks.py          # Capture static gestures
├── capture_sequence.py           # Capture motion sequences
├── prepare_data.py               # Process static data
├── prepare_seq_dataset.py        # Process sequence data
├── train_mlp.py                  # Train static model
├── train_lstm.py                 # Train motion model
├── requirements.txt              # Dependencies
└── models/                       # Trained models (not included)
```

## 🧠 Model Performance

- **Inference Speed**: ~20ms per frame
- **Real-time FPS**: 25-30 FPS
- **Latency**: Real-time detection with minimal delay
- **Reliability**: Stable predictions with majority voting smoothing

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Reyansh Kalra**
- GitHub: [@reyanshlakra2011-coder](https://github.com/reyanshlakra2011-coder)
- Project built as part of school exhibition 2025

## 🙏 Acknowledgments

- Computer Science teacher for guidance and support
- MediaPipe team for hand tracking library
- TensorFlow team for deep learning framework
- School exhibition platform for opportunity

## 📧 Contact

For questions or collaboration opportunities, feel free to reach out!

---

**Note**: This project was built by a Class 9 student passionate about AI and accessibility. The goal is to make communication easier for the deaf and mute community.

**⭐ If you found this project useful, please give it a star!**

https://github.com/reyanshlakra2011-coder/AI-sign-language-translator
