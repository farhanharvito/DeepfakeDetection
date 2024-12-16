# Deepfake Detection Application 🕵️‍♀️🤖

## Project Overview

This project implements deepfake detection system using a combination of Xception neural network architecture and LSTM (Long Short-Term Memory) networks. The application provides a user-friendly Streamlit interface for real-time deepfake image detection.

## 🌟 Key Features

- **Advanced Deep Learning Model**: Utilizes Xception as the base feature extractor
- **Temporal Analysis**: Incorporates LSTM for enhanced temporal feature recognition
- **High Accuracy Deepfake Detection**: Robust model trained on diverse deepfake datasets

## 🛠 Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Web Interface**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Image Processing**: OpenCV
- **Visualization**: Matplotlib, Seaborn

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥 Running the Application

To launch the Streamlit application:

```bash
streamlit run app.py
```

## 📊 Model Performance

### Metrics
- **Accuracy**: 91.67%
- **Precision**: 89.06%
- **Recall**: 95%
- **F1 Score**: 91.64%

## 🔍 How It Works

1. **Feature Extraction**: 
   - Xception network extracts deep visual features from input images
   - Handles complex spatial patterns in potential deepfake images

2. **Temporal Analysis**:
   - LSTM layers process extracted features
   - Captures temporal dependencies and sequence-level information

3. **Classification**:
   - Final dense layers make binary classification (Deepfake vs. Real)

## 📋 Requirements

See `requirements.txt` for the complete list of dependencies.

## 🧾 License

Distributed under the MIT License. See `LICENSE` for more information.


## 🔬 Research and References

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [A Comparative Study of Deepfake Video Detection Method]([link-to-relevant-paper](https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-ICOIACT-2020.pdf))

**Disclaimer**: This project is for educational and research purposes. Always use AI responsibly.
