"""
app.py
------
Streamlit web app for real-time sign language recognition.
Uses browser camera (works locally AND on Streamlit Cloud).
Captures a photo, detects hand landmarks, predicts the sign.

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import joblib
import os
import time
from datetime import datetime
from io import BytesIO

# pyttsx3 is optional (only works locally, not on cloud)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Hand landmark connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm
]

# Page configuration
st.set_page_config(
    page_title="SignSpeak - Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────── */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0;
        max-width: 1200px;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu, footer {visibility: hidden;}

    /* ── Brand header ───────────────────────────── */
    .brand-banner {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .brand-banner .icon {
        font-size: 3rem;
        line-height: 1;
    }
    .brand-banner .text h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .brand-banner .text p {
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
        color: rgba(255,255,255,0.55);
        font-weight: 400;
    }

    /* ── Stat cards ─────────────────────────────── */
    .stat-card {
        background: #0e1117;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .stat-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.45);
        margin-bottom: 0.3rem;
    }
    .stat-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-card .value.green { color: #22c55e; }
    .stat-card .value.blue  { color: #3b82f6; }
    .stat-card .value.amber { color: #f59e0b; }

    /* ── Prediction badge ───────────────────────── */
    .prediction-badge {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .prediction-badge .sign-label {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .prediction-badge .confidence {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.25rem;
    }
    .no-prediction {
        background: #0e1117;
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        color: rgba(255,255,255,0.35);
        font-size: 0.95rem;
    }

    /* ── History table ──────────────────────────── */
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
    .history-row:nth-child(odd)  { background: rgba(255,255,255,0.02); }
    .history-row:nth-child(even) { background: rgba(255,255,255,0.04); }
    .history-row .time   { color: rgba(255,255,255,0.4); width: 25%; }
    .history-row .sign   { color: #ffffff; font-weight: 600; width: 50%; }
    .history-row .conf   { color: #22c55e; font-weight: 500; width: 25%; text-align: right; }

    /* ── Instruction callout ────────────────────── */
    .instruction-callout {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }
    .instruction-callout .icon { font-size: 1.3rem; }
    .instruction-callout .body {
        color: rgba(255,255,255,0.7);
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .instruction-callout .body strong {
        color: #ffffff;
    }

    /* ── Sidebar tweaks ─────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0e1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stSlider > label,
    section[data-testid="stSidebar"] .stCheckbox > label {
        font-size: 0.88rem;
    }
    .sidebar-section-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255,255,255,0.3);
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .model-info-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
        line-height: 1.8;
    }

    /* ── Result card ────────────────────────────── */
    .result-card {
        background: #0e1117;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        overflow: hidden;
    }
    .result-card .image-section {
        position: relative;
    }
    .result-card .info-section {
        padding: 1rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_FILE = "sign_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
CONFIDENCE_THRESHOLD = 0.7

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_spoken' not in st.session_state:
    st.session_state.last_spoken = None
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

# ── Cached loaders (only load once per session) ──────────

@st.cache_resource
def load_hand_landmarker():
    """Load MediaPipe hand landmarker model (cached)."""
    base_options = mp_python.BaseOptions(
        model_asset_path='hand_landmarker.task'
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_vision.HandLandmarker.create_from_options(options)

@st.cache_resource
def load_ml_model():
    """Load trained ML model, scaler, and label encoder (cached)."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None, None

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    label_encoder = None
    if os.path.exists(LABEL_ENCODER_FILE):
        label_encoder = joblib.load(LABEL_ENCODER_FILE)

    return model, scaler, label_encoder

# ── Core functions ────────────────────────────────────────

def process_image(image_bytes, landmarker):
    """
    Process a camera capture: decode image, detect landmarks, draw overlay.
    Returns (landmarks_array, annotated_frame_bgr, hand_detected).
    """
    # Decode image from bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return None, None, False

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Run MediaPipe detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    hand_detected = len(results.hand_landmarks) > 0

    if hand_detected:
        h, w, _ = frame.shape
        for hand_landmarks in results.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], (0, 200, 120), 2)
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)

        # Extract features from first hand
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks).reshape(1, -1), frame, True

    return None, frame, False

def predict_sign(landmarks, model, scaler, label_encoder=None):
    """Predict sign label and confidence from landmarks."""
    if landmarks is None:
        return None, 0.0

    landmarks_scaled = scaler.transform(landmarks)
    probabilities = model.predict_proba(landmarks_scaled)[0]
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]

    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    else:
        predicted_label = model.classes_[predicted_idx]

    return predicted_label, confidence

def draw_overlay(frame, text, color):
    """Draw text overlay with background for readability."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.rectangle(frame, (8, 35 - th - 8), (18 + tw, 35 + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

# ── UI components ─────────────────────────────────────────

def render_brand_header():
    st.markdown("""
    <div class="brand-banner">
        <div class="icon">🤟</div>
        <div class="text">
            <h1>SignSpeak</h1>
            <p>Real-time sign language translation powered by machine learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(model, label_encoder):
    """Render sidebar and return confidence_threshold."""
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Detection Settings</div>',
        unsafe_allow_html=True
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.30,
        max_value=0.95,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence level required to register a prediction"
    )

    st.sidebar.markdown(
        '<div class="sidebar-section-title">Actions</div>',
        unsafe_allow_html=True
    )

    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.detection_count = 0

    st.sidebar.markdown(
        '<div class="sidebar-section-title">Model</div>',
        unsafe_allow_html=True
    )

    model_type = type(model).__name__
    model_names = {
        'KNeighborsClassifier': 'K-Nearest Neighbors',
        'MLPClassifier': 'Neural Network (MLP)',
        'RandomForestClassifier': 'Random Forest'
    }
    display_name = model_names.get(model_type, model_type)

    classes = label_encoder.classes_ if label_encoder is not None else model.classes_
    alpha = [c for c in classes if len(c) == 1 and c.isalpha()]
    digits = [c for c in classes if c.isdigit()]
    words = [c for c in classes if len(c) > 1 or (len(c) == 1 and not c.isalpha())]

    labels_html = ""
    if alpha:
        labels_html += f"<div><strong style='color:rgba(255,255,255,0.75)'>Letters:</strong> {' '.join(alpha)}</div>"
    if digits:
        labels_html += f"<div><strong style='color:rgba(255,255,255,0.75)'>Numbers:</strong> {' '.join(digits)}</div>"
    if words:
        labels_html += f"<div><strong style='color:rgba(255,255,255,0.75)'>Words:</strong> {', '.join(words)}</div>"

    st.sidebar.markdown(f"""
    <div class="model-info-card">
        <div><strong style="color:rgba(255,255,255,0.75)">Type:</strong> {display_name}</div>
        <div><strong style="color:rgba(255,255,255,0.75)">Signs:</strong> {len(classes)} classes</div>
        {labels_html}
    </div>
    """, unsafe_allow_html=True)

    return confidence_threshold

def render_stat_cards(history):
    if len(history) > 0:
        unique_signs = len(set(h['label'] for h in history))
        total = len(history)
        avg_conf = np.mean([h['confidence'] for h in history])
    else:
        unique_signs = 0
        total = 0
        avg_conf = 0.0

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Total Detections</div>
            <div class="value blue">{total}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Unique Signs</div>
            <div class="value amber">{unique_signs}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Avg Confidence</div>
            <div class="value green">{avg_conf:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

def render_prediction_panel(history):
    if len(history) > 0:
        latest = history[-1]
        st.markdown(f"""
        <div class="prediction-badge">
            <div class="sign-label">{latest['label']}</div>
            <div class="confidence">{latest['confidence']:.0%} confidence &middot; {latest['time']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-prediction">
            No prediction yet.<br/>Take a photo with the camera and click capture.
        </div>
        """, unsafe_allow_html=True)

def render_history_list(history):
    if len(history) == 0:
        st.markdown("""
        <div style="color:rgba(255,255,255,0.3); font-size:0.85rem; text-align:center; padding:1rem 0;">
            Recognition history will appear here
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <div style="display:flex; justify-content:space-between; padding:0.4rem 0.8rem;
                font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
                color:rgba(255,255,255,0.3); border-bottom:1px solid rgba(255,255,255,0.06);">
        <span style="width:25%">Time</span>
        <span style="width:50%">Sign</span>
        <span style="width:25%; text-align:right">Confidence</span>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for entry in reversed(history[-20:]):
        rows_html += f"""
        <div class="history-row">
            <span class="time">{entry['time']}</span>
            <span class="sign">{entry['label'].upper()}</span>
            <span class="conf">{entry['confidence']:.0%}</span>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────

def main():
    render_brand_header()

    # Load models (cached across reruns)
    model, scaler, label_encoder = load_ml_model()
    if model is None:
        st.error("Model files not found! Please train the model first.")
        st.info("Run locally:  `python train_model.py`")
        st.stop()

    landmarker = load_hand_landmarker()

    # Sidebar
    confidence_threshold = render_sidebar(model, label_encoder)

    # ── Main layout ──────────────────────────────────────
    col_feed, col_panel = st.columns([3, 2], gap="large")

    # ── LEFT COLUMN: Camera ──────────────────────────────
    with col_feed:
        st.markdown("#### Camera Capture")

        # Browser-based camera (works on Streamlit Cloud + local)
        camera_image = st.camera_input(
            "Take a photo of your hand sign",
            help="Position your hand clearly in the frame and click the capture button"
        )

        if camera_image is not None:
            # Process the captured image
            landmarks, annotated_frame, hand_detected = process_image(
                camera_image, landmarker
            )

            if annotated_frame is not None:
                # Predict sign if hand detected
                if hand_detected and landmarks is not None:
                    predicted_label, confidence = predict_sign(
                        landmarks, model, scaler, label_encoder
                    )
                    st.session_state.detection_count += 1

                    if confidence > confidence_threshold:
                        draw_overlay(
                            annotated_frame,
                            f"{predicted_label.upper()}  {confidence:.0%}",
                            (0, 255, 100)
                        )

                        timestamp = datetime.now().strftime("%H:%M:%S")
                        if (len(st.session_state.history) == 0 or
                                st.session_state.history[-1]['label'] != predicted_label):
                            st.session_state.history.append({
                                'label': predicted_label,
                                'confidence': confidence,
                                'time': timestamp
                            })
                    else:
                        draw_overlay(
                            annotated_frame,
                            f"Low confidence: {confidence:.0%}",
                            (100, 180, 255)
                        )
                else:
                    draw_overlay(
                        annotated_frame,
                        "No hand detected - try again",
                        (180, 180, 180)
                    )

                # Display annotated frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_column_width=True)

        else:
            # Idle state
            st.markdown("""
            <div style="
                background: #0e1117;
                border: 1px dashed rgba(255,255,255,0.1);
                border-radius: 16px;
                padding: 4rem 2rem;
                text-align: center;
                color: rgba(255,255,255,0.3);
            ">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">📷</div>
                <div style="font-size:1rem; font-weight:500; color:rgba(255,255,255,0.5);">
                    Camera ready
                </div>
                <div style="font-size:0.85rem; margin-top:0.3rem;">
                    Click the camera button above to capture a sign
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Instructions
        st.markdown("")
        classes = label_encoder.classes_ if label_encoder is not None else model.classes_
        alpha = sorted([c for c in classes if len(c) == 1 and c.isalpha()])
        digits = sorted([c for c in classes if c.isdigit()])
        words = sorted([c for c in classes if len(c) > 1 or (len(c) == 1 and not c.isalpha())])

        sign_list_parts = []
        if alpha:
            sign_list_parts.append(f"<strong>Alphabet:</strong> {' '.join(alpha)}")
        if digits:
            sign_list_parts.append(f"<strong>Numbers:</strong> {' '.join(digits)}")
        if words:
            sign_list_parts.append(f"<strong>Words:</strong> {', '.join(words)}")

        sign_list_html = " &nbsp;|&nbsp; ".join(sign_list_parts)

        st.markdown(f"""
        <div class="instruction-callout">
            <div class="icon">💡</div>
            <div class="body">
                <strong>How to use:</strong> Click the camera button, position your hand clearly
                in the frame, and take a photo. The model recognizes
                <strong>{len(classes)} signs</strong>:<br/>
                {sign_list_html}<br/><br/>
                Adjust the <strong>confidence threshold</strong> in the sidebar if predictions
                seem off. Take multiple photos to build up recognition history.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── RIGHT COLUMN: Prediction + History + Stats ───────
    with col_panel:
        st.markdown("#### Current Prediction")
        render_prediction_panel(st.session_state.history)

        st.markdown("")

        st.markdown("#### Session Statistics")
        render_stat_cards(st.session_state.history)

        st.markdown("")

        st.markdown("#### Recognition History")
        render_history_list(st.session_state.history)

if __name__ == "__main__":
    main()
