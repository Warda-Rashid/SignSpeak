# SignSpeak: Real-Time Sign Language Translator

A real-time sign language to text and speech translator using computer vision and machine learning. This project uses MediaPipe for hand landmark detection and scikit-learn for sign classification.

## Features

- **Real-time hand landmark detection** using MediaPipe Hands (21 landmarks per hand)
- **Machine learning classification** with KNN or MLP classifier
- **Live webcam feed** with Streamlit web interface
- **Text-to-speech** output for recognized signs (optional)
- **Confidence filtering** to avoid false positives
- **Recognition history** with statistics tracking

## Supported Sign Labels

The system supports the following 15 sign language gestures:
- hello, thanks, yes, no, please
- sorry, help, stop, name, good
- bad, water, food, more, done

## Project Structure

```
signspeak/
├── data_collection.py    # Collect training samples from webcam
├── train_model.py        # Train ML model on collected data
├── app.py                # Streamlit web app for live recognition
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

Run the data collection script to gather samples for each sign:

```bash
python data_collection.py
```

**Controls:**
- **SPACE** - Capture a sample for the current label
- **'n'** - Move to the next label
- **'q'** - Quit

**Tips for better data:**
- Collect at least 20-30 samples per sign
- Vary your hand position slightly between samples
- Ensure good lighting
- Keep your hand fully visible in the frame

The samples are saved to `sign_data.csv` with 63 features (21 landmarks × 3 coordinates) plus the label.

### 3. Train the Model

```bash
python train_model.py
```

The training script will:
- Load the collected data
- Split into train/test sets (80/20)
- Train a KNN classifier (tries k=1 to k=10)
- If KNN accuracy < 80%, also trains an MLP neural network
- Automatically selects the best performing model
- Saves the model to `sign_model.pkl` and scaler to `scaler.pkl`

### 4. Run the Live Demo

```bash
streamlit run app.py
```

The web app will open in your browser with:
- **Left panel**: Live webcam feed with landmark visualization
- **Right panel**: Current prediction, history log, and statistics
- **Sidebar controls**: 
  - Confidence threshold slider (default: 0.7)
  - Speech enable/disable toggle
  - Clear history button

## How It Works

### Data Collection (`data_collection.py`)
1. Opens webcam and runs MediaPipe Hands for real-time landmark detection
2. When you press SPACE, extracts 21 hand landmarks (x, y, z coordinates)
3. Flattens the 63 coordinates into a feature vector
4. Saves the vector with the current label to CSV

### Model Training (`train_model.py`)
1. Loads CSV data and splits into features (landmarks) and labels
2. Applies StandardScaler to normalize features
3. Trains KNN with different k values to find optimal k
4. If KNN accuracy is low, trains MLP as alternative
5. Evaluates on test set and saves best model

### Live Recognition (`app.py`)
1. Streamlit app captures webcam frames in real-time
2. MediaPipe extracts hand landmarks from each frame
3. Landmarks are scaled and fed to the trained model
4. If confidence > threshold, displays and optionally speaks the prediction
5. Maintains a history log of all recognized signs

## Usage Tips

- **Lighting**: Ensure consistent, good lighting for better landmark detection
- **Background**: Use a plain background to improve hand detection
- **Hand Position**: Keep your hand fully visible and centered in the frame
- **Confidence Threshold**: Adjust in the sidebar if you get too many false positives or miss valid signs
- **Training Data**: More diverse samples = better accuracy. Collect samples from different angles and positions

## Troubleshooting

**"No hand detected" message:**
- Make sure your hand is fully visible in the camera frame
- Check lighting conditions
- Move your hand closer to or further from the camera

**Low accuracy:**
- Collect more training samples (aim for 30+ per sign)
- Ensure training samples are diverse (different angles, positions)
- Retrain the model with `python train_model.py`

**Webcam not opening:**
- Check if another application is using the camera
- On Linux, ensure you have proper permissions: `sudo usermod -a -G video $USER`
- Try a different camera index in the code (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

**Speech not working:**
- Install pyttsx3 dependencies:
  - Windows: Should work out of the box
  - Linux: `sudo apt-get install espeak ffmpeg libespeak1`
  - macOS: Should work with built-in speech synthesis

## Requirements

- Python 3.8+
- Webcam
- See `requirements.txt` for all Python packages

## Future Improvements

- Add more sign labels
- Support two-handed signs
- Implement continuous sign recognition (sign sequences)
- Add data augmentation for better generalization
- Deploy as a web service
- Add support for different languages

## License

This is an educational project for learning computer vision and machine learning concepts.

## Credits

Built with:
- [MediaPipe](https://mediapipe.dev/) - Hand landmark detection
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Streamlit](https://streamlit.io/) - Web interface
- [OpenCV](https://opencv.org/) - Computer vision
- [pyttsx3](https://pyttsx3.readthedocs.io/) - Text-to-speech
