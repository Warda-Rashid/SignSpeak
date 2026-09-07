"""
app.py
------
SignSpeak live sign-language recognition web app.

Two modes:
  - Live Camera  : continuous webcam stream via streamlit-webrtc (getUserMedia)
                   with real-time MediaPipe landmark detection + ML prediction
                   + stability filtering + deduplicated sign history.
  - Photo Capture: single-shot fallback (uses st.camera_input).

Usage:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import joblib
import os
import time
import threading
from collections import deque
from datetime import datetime

# Live camera + auto-refresh
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh

# Optional text-to-speech (only works locally)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

MODEL_FILE = "sign_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES = 8
ABSENCE_COOLDOWN = 2.0  # seconds before same sign can be re-recorded after absence

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignSpeak - Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS: explicit dark backgrounds so elements remain visible
# regardless of the active Streamlit theme, plus responsive
# breakpoints for mobile/tablet.
# ─────────────────────────────────────────────────────────
STYLES = """
<style>
/* ── Global ─────────────────────────────────── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 0;
    max-width: 1200px;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── Brand banner ───────────────────────────── */
.brand-banner {
    background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    border: 1px solid rgba(255,255,255,0.06);
}
.brand-banner .icon { font-size: 3rem; line-height: 1; }
.brand-banner .text h1 {
    margin: 0; font-size: 1.8rem; font-weight: 700;
    color: #ffffff; letter-spacing: -0.02em;
}
.brand-banner .text p {
    margin: 0.25rem 0 0 0; font-size: 0.95rem;
    color: rgba(255,255,255,0.55); font-weight: 400;
}

/* ── Stat cards ──────────────────────────────── */
.stat-card {
    background: #0e1117;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.stat-card .label {
    font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: rgba(255,255,255,0.45);
    margin-bottom: 0.3rem;
}
.stat-card .value { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
.stat-card .value.green { color: #22c55e; }
.stat-card .value.blue  { color: #3b82f6; }
.stat-card .value.amber { color: #f59e0b; }

/* ── Prediction badge ────────────────────────── */
.prediction-badge {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    border-radius: 14px; padding: 1.5rem 2rem;
    text-align: center; margin-bottom: 0.5rem;
}
.prediction-badge .sign-label {
    font-size: 2.2rem; font-weight: 800;
    color: #ffffff; letter-spacing: 0.04em; text-transform: uppercase;
}
.prediction-badge .confidence {
    font-size: 0.85rem; color: rgba(255,255,255,0.8); margin-top: 0.25rem;
}
.no-prediction {
    background: #0e1117;
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 14px; padding: 1.5rem 2rem;
    text-align: center; color: rgba(255,255,255,0.4);
    font-size: 0.95rem;
}

/* ── History ─────────────────────────────────── */
.history-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0.8rem; border-radius: 8px; margin-bottom: 2px;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.history-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0.8rem; border-radius: 8px; margin-bottom: 2px;
    font-size: 0.85rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.04);
}
.history-row:nth-child(odd)  { background: rgba(255,255,255,0.05); }
.history-row:nth-child(even) { background: rgba(255,255,255,0.02); }
.history-row .time { color: rgba(255,255,255,0.5); width: 25%; }
.history-row .sign { color: #ffffff; font-weight: 600; width: 50%; }
.history-row .conf { color: #22c55e; font-weight: 500; width: 25%; text-align: right; }

/* ── Status panel ────────────────────────────── */
.status-box {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-weight: 600;
    font-size: 0.95rem;
    border-left: 4px solid;
    margin-bottom: 0.75rem;
    background: #0e1117;
    color: #fafafa;
}

/* ── Stability progress ──────────────────────── */
.stability-track {
    background: rgba(255,255,255,0.08);
    border-radius: 6px; height: 8px; width: 100%;
    margin-top: 0.5rem; overflow: hidden;
}
.stability-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #22c55e 100%);
    transition: width 0.15s ease-out;
}

/* ── Camera placeholder ─────────────────────── */
.camera-idle {
    background: #0e1117;
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    color: rgba(255,255,255,0.4);
}
.camera-idle .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }

/* ── Camera start placeholder ────────────────── */
.camera-start-placeholder {
    background: #0e1117;
    border: 2px dashed rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    color: rgba(255,255,255,0.5);
    aspect-ratio: 4/3;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.camera-start-placeholder .icon { font-size: 3rem; margin-bottom: 0.8rem; }
.camera-start-placeholder .title {
    font-size: 1.1rem; font-weight: 600;
    color: rgba(255,255,255,0.7); margin-bottom: 0.3rem;
}
.camera-start-placeholder .hint {
    font-size: 0.85rem; color: rgba(255,255,255,0.4);
}

/* ── Instruction callout ─────────────────────── */
.instruction-callout {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex; gap: 1rem; align-items: flex-start;
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem; line-height: 1.6;
}
.instruction-callout .icon { font-size: 1.3rem; }
.instruction-callout strong { color: #ffffff; }

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0e1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-section-title {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: rgba(255,255,255,0.4);
    margin: 1.2rem 0 0.5rem 0; padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.model-info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.6);
    line-height: 1.8;
}

/* ── Camera video wrapper ────────────────────── */
.webrtc-wrapper {
    width: 100%;
    aspect-ratio: 4/3;
    position: relative;
    background: #000;
    border-radius: 12px;
    overflow: hidden;
}
.webrtc-wrapper video {
    width: 100% !important;
    height: 100% !important;
    max-height: 65vh;
    object-fit: cover;
    border-radius: 12px;
    background: #000;
}

/* ── Right panel compact ─────────────────────── */
.panel-tight h4 { margin-bottom: 0.35rem !important; margin-top: 0 !important; }
.panel-tight .status-box { margin-bottom: 0.3rem !important; }
.panel-tight .prediction-badge { margin-bottom: 0.2rem !important; padding: 1rem 1.5rem !important; }
.panel-tight .no-prediction { padding: 1rem 1.5rem !important; }

/* ── Responsive: stack columns on narrow screens */
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    .brand-banner .text h1 { font-size: 1.4rem; }
    .stat-card .value { font-size: 1.3rem; }
    .prediction-badge .sign-label { font-size: 1.6rem; }
    .stat-card { padding: 0.9rem; }
    .webrtc-wrapper { aspect-ratio: 16/10; }
    .camera-start-placeholder { aspect-ratio: 16/10; padding: 2rem 1rem; }
    .camera-start-placeholder .title { font-size: 1rem; }
}
@media (max-width: 480px) {
    .brand-banner { padding: 1.2rem; }
    .brand-banner .icon { font-size: 2rem; }
    .brand-banner .text h1 { font-size: 1.15rem; }
    .stat-card { padding: 0.7rem; }
    .stat-card .value { font-size: 1.1rem; }
    .sidebar-section-title { margin: 0.8rem 0 0.3rem 0; }
    .instruction-callout { padding: 0.8rem 1rem; font-size: 0.82rem; }
    .prediction-badge { padding: 0.8rem 1rem; }
    .prediction-badge .sign-label { font-size: 1.3rem; }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# JavaScript to aggressively release camera streams on page load
CAMERA_RELEASE_JS = """
<script>
(function() {
    // Immediately stop ALL active media streams in this page
    function killAllStreams() {
        document.querySelectorAll('video, audio').forEach(el => {
            if (el.srcObject) {
                try {
                    el.srcObject.getTracks().forEach(t => t.stop());
                    el.srcObject = null;
                } catch(e) { console.log('stream cleanup:', e); }
            }
        });
        // Also try getUserMedia to grab and immediately release (clears stale handles)
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({video: true, audio: false})
                .then(stream => {
                    stream.getTracks().forEach(t => t.stop());
                    console.log('Camera probe+release successful');
                })
                .catch(err => console.log('Camera probe skipped:', err.message));
        }
    }
    // Run immediately and again after a short delay (catches late-loading iframes)
    killAllStreams();
    setTimeout(killAllStreams, 800);

    // Expose globally so the Release Camera button can call it
    window.releaseAllCameras = killAllStreams;
})();
</script>
"""
components.html(CAMERA_RELEASE_JS, height=0)


# ─────────────────────────────────────────────────────────
# Recognition state (survives reruns via st.session_state)
# ─────────────────────────────────────────────────────────
class RecognitionState:
    """Thread-safe shared state for live recognition pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        # History
        self.history: list[dict] = []
        # Stability buffer
        self.stability_deque: deque = deque(maxlen=STABLE_FRAMES)
        # Dedup
        self.last_saved_label: str | None = None
        self.last_absent_at: float = 0.0
        # Live UI state (read by UI, written by callback)
        self.current_status = "ready"
        self.current_label: str | None = None
        self.current_confidence: float = 0.0
        self.stability_progress: float = 0.0
        # Timestamp counter for MediaPipe VIDEO mode (ms, monotonic)
        self.frame_ts_ms: int = 0
        # Config (updated from UI each rerun)
        self.threshold: float = CONFIDENCE_THRESHOLD
        self.stable_frames: int = STABLE_FRAMES
        self.speak_enabled: bool = False


# ─────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_hand_landmarker():
    """MediaPipe hand landmarker (VIDEO mode for smooth tracking)."""
    base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Load trained ML model, scaler, and label encoder."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None, None
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    le = None
    if os.path.exists(LABEL_ENCODER_FILE):
        le = joblib.load(LABEL_ENCODER_FILE)
    return model, scaler, le


# ─────────────────────────────────────────────────────────
# Frame drawing helpers
# ─────────────────────────────────────────────────────────
def _draw_overlay(frame, text, color=(0, 255, 100), y=35, scale=0.85, thickness=2):
    """Draw text with solid dark background for readability on the video frame."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 10
    # Solid background rectangle
    cv2.rectangle(frame, (8, y - th - pad), (18 + tw, y + pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_stability_bar(frame, progress, y_offset=0):
    """Draw a stability progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_y = h - 35 + y_offset
    bar_w = min(300, w - 40)
    bar_h = 10
    x0 = (w - bar_w) // 2
    # Track
    cv2.rectangle(frame, (x0, bar_y), (x0 + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    # Fill
    fill_w = int(bar_w * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (x0, bar_y), (x0 + fill_w, bar_y + bar_h), (0, 200, 100), -1)
    # Label
    label = f"Stability: {progress:.0%}"
    cv2.putText(frame, label, (x0, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)


def _draw_hand_skeleton(frame, hand_landmarks):
    """Draw hand skeleton overlay."""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, points[s], points[e], (0, 200, 120), 2, cv2.LINE_AA)
    for p in points:
        cv2.circle(frame, p, 5, (0, 255, 0), -1)
        cv2.circle(frame, p, 5, (255, 255, 255), 1)


# ─────────────────────────────────────────────────────────
# Text-to-speech helper
# ─────────────────────────────────────────────────────────
_speak_lock = threading.Lock()


def speak_sign(label):
    """Speak a sign label asynchronously. Non-blocking; drops if already speaking."""
    if not TTS_AVAILABLE:
        return
    if not _speak_lock.acquire(blocking=False):
        return

    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(label.replace("_", " "))
            engine.runAndWait()
        except Exception:
            pass
        finally:
            _speak_lock.release()

    threading.Thread(target=_speak, daemon=True).start()


# ─────────────────────────────────────────────────────────
# Video frame callback (core recognition pipeline)
# ─────────────────────────────────────────────────────────
def make_video_frame_callback(state, landmarker, model, scaler, label_encoder):
    """Return a frame callback that runs the full recognition pipeline.

    The callback is recreated each Streamlit rerun so it always sees the
    latest session state and configuration values.
    """

    def video_frame_callback(frame):
        try:
            # Convert incoming av.VideoFrame → numpy BGR
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # mirror effect
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Monotonic timestamp (ms) for VIDEO mode tracking
            with state.lock:
                state.frame_ts_ms += 33
                ts = state.frame_ts_ms
            results = landmarker.detect_for_video(mp_image, ts)

            hand_detected = len(results.hand_landmarks) > 0

            if hand_detected:
                # Draw skeleton
                for hl in results.hand_landmarks:
                    _draw_hand_skeleton(img, hl)

                # Extract 63 features (21 landmarks × x, y, z)
                hand_lms = results.hand_landmarks[0]
                feats = np.array(
                    [c for lm in hand_lms for c in (lm.x, lm.y, lm.z)],
                    dtype=np.float64,
                ).reshape(1, -1)

                # Predict
                feats_scaled = scaler.transform(feats)
                probs = model.predict_proba(feats_scaled)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                enc_class = model.classes_[idx]
                label = label_encoder.inverse_transform([enc_class])[0] if label_encoder else str(enc_class)

                # Stability tracking
                now = time.time()
                with state.lock:
                    state.stability_deque.append((label, conf))
                    threshold = state.threshold
                    stable_n = state.stable_frames
                    buf = list(state.stability_deque)

                    if len(buf) == stable_n:
                        labels = [x[0] for x in buf]
                        confs = [x[1] for x in buf]
                        # Stable when ALL frames have the same label and all confs >= threshold
                        all_same = all(lb == labels[0] for lb in labels)
                        all_conf_ok = all(c >= threshold for c in confs)

                        if all_same and all_conf_ok:
                            accepted_label = labels[0]
                            accepted_conf = float(np.mean(confs))

                            # Dedup logic: save only when different from last saved
                            # OR same sign after absence cooldown has passed
                            can_save = (
                                state.last_saved_label is None
                                or accepted_label != state.last_saved_label
                                or (now - state.last_absent_at >= ABSENCE_COOLDOWN)
                            )
                            if can_save:
                                state.history.append({
                                    "label": accepted_label,
                                    "confidence": accepted_conf,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                })
                                state.last_saved_label = accepted_label
                                state.last_absent_at = now
                                if state.speak_enabled:
                                    speak_sign(accepted_label)

                            state.current_label = accepted_label
                            state.current_confidence = accepted_conf
                            state.current_status = "sign_detected"
                            state.stability_progress = 1.0
                        else:
                            # Not yet stable — show current prediction with progress
                            state.current_label = label
                            state.current_confidence = conf
                            # Count trailing consecutive same-label frames
                            run = 1
                            for i in range(len(labels) - 2, -1, -1):
                                if labels[i] == label:
                                    run += 1
                                else:
                                    break
                            state.stability_progress = run / stable_n
                            if conf < threshold:
                                state.current_status = "low_confidence"
                            else:
                                state.current_status = "recognizing"
                    else:
                        # Buffer filling
                        state.current_label = label
                        state.current_confidence = conf
                        state.stability_progress = len(buf) / stable_n
                        state.current_status = "recognizing"

                    # Mark hand present
                    state.last_absent_at = now

                # Annotate frame
                if state.current_status == "sign_detected":
                    _draw_overlay(img, f"{label.upper()}  {conf:.0%}", (0, 255, 100))
                elif state.current_status == "low_confidence":
                    _draw_overlay(img, f"Low conf: {conf:.0%}", (100, 180, 255))
                else:
                    _draw_overlay(img, f"{label}  {conf:.0%}", (220, 220, 220))
                _draw_stability_bar(img, state.stability_progress)

            else:
                # No hand detected
                with state.lock:
                    now = time.time()
                    # Update absence timestamp only when hand was previously present
                    if state.current_status != "no_hand":
                        state.last_absent_at = now
                    state.stability_deque.clear()
                    state.current_status = "no_hand"
                    state.current_label = None
                    state.current_confidence = 0.0
                    state.stability_progress = 0.0
                _draw_overlay(img, "No hand — show your hand", (180, 180, 180))

        except Exception as exc:
            # Graceful degradation — never crash the frame pipeline
            _draw_overlay(img, "Error processing frame", (100, 100, 255))

        import av
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return video_frame_callback


# ─────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────
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


def render_sidebar(model, label_encoder, state):
    """Render sidebar controls and update state config. Returns nothing."""
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Detection</div>',
        unsafe_allow_html=True,
    )

    state.threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.30, max_value=0.95, value=CONFIDENCE_THRESHOLD, step=0.05,
        help="Minimum confidence required to accept a sign prediction.",
    )
    state.stable_frames = st.sidebar.slider(
        "Stability Frames",
        min_value=3, max_value=15, value=STABLE_FRAMES, step=1,
        help="Number of consecutive matching frames required before a sign is accepted.",
    )

    if TTS_AVAILABLE:
        state.speak_enabled = st.sidebar.checkbox(
            "Speak recognized signs",
            value=False,
            help="Read each accepted sign aloud via text-to-speech.",
        )

    st.sidebar.markdown(
        '<div class="sidebar-section-title">Actions</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Clear History", use_container_width=True):
        with state.lock:
            state.history.clear()
            state.last_saved_label = None
            state.last_absent_at = 0.0

    if st.sidebar.button("Release Camera", use_container_width=True,
                         help="Force release the camera if it's stuck or showing 'Device in use' error"):
        # Reset Streamlit-side camera state
        st.session_state.camera_in_use = False
        st.session_state.camera_start_requested = False
        st.session_state.camera_start_error = None
        # Inject JS to release camera and trigger refresh
        components.html("""
        <script>
        // Aggressively stop all video tracks in every iframe
        function killAll() {
            document.querySelectorAll('video, audio').forEach(el => {
                if (el.srcObject) {
                    el.srcObject.getTracks().forEach(t => t.stop());
                    el.srcObject = null;
                }
            });
            // Also probe+release to clear stale OS-level handles
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({video:true,audio:false})
                    .then(s => { s.getTracks().forEach(t=>t.stop()); })
                    .catch(()=>{});
            }
        }
        killAll();
        // Also run in parent document
        try { window.parent.releaseAllCameras && window.parent.releaseAllCameras(); } catch(e) {}
        setTimeout(() => { window.parent.location.reload(); }, 600);
        </script>
        """, height=0)
        st.sidebar.success("Camera released! Refreshing page...")

    st.sidebar.markdown(
        '<div class="sidebar-section-title">Model</div>',
        unsafe_allow_html=True,
    )

    model_names = {
        "KNeighborsClassifier": "K-Nearest Neighbors",
        "MLPClassifier": "Neural Network (MLP)",
        "RandomForestClassifier": "Random Forest",
    }
    display_name = model_names.get(type(model).__name__, type(model).__name__)
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


def render_status_panel(state):
    """Show current recognition status with a colored indicator."""
    with state.lock:
        status = state.current_status
        label = state.current_label
        conf = state.current_confidence
        progress = state.stability_progress

    styles = {
        "ready":         ("#64748b", "📷 Camera ready — show a sign"),
        "starting":      ("#64748b", "⏳ Camera starting…"),
        "no_hand":       ("#64748b", "🖐  No hand detected — show your hand"),
        "recognizing":   ("#3b82f6", f"🔍 Recognizing…  {label or ''}  ({conf:.0%})"),
        "sign_detected": ("#22c55e", f"✅ Sign detected: {label.upper() if label else '?'}  ({conf:.0%})"),
        "low_confidence":("#f59e0b", f"⚠️  Low confidence — hold your sign steady"),
        "camera_error":  ("#ef4444", "❌ Camera error"),
        "stopped":       ("#64748b", "⏹  Camera stopped"),
    }
    color, text = styles.get(status, ("#64748b", status))
    st.markdown(f"""
    <div class="status-box" style="border-left-color: {color}; background: {color}15;">
        <div style="color: {color};">{text}</div>
        {"" if status != "recognizing" else f'''
        <div class="stability-track">
            <div class="stability-fill" style="width: {progress*100:.0f}%;"></div>
        </div>
        <div style="color: rgba(255,255,255,0.5); font-size:0.75rem; margin-top:4px;">
            Stability: {progress:.0%} — hold steady
        </div>
        '''}
    </div>
    """, unsafe_allow_html=True)


def render_stat_cards(history):
    if history:
        unique = len(set(h["label"] for h in history))
        total = len(history)
        avg = float(np.mean([h["confidence"] for h in history]))
    else:
        unique = total = 0
        avg = 0.0

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
            <div class="value amber">{unique}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Avg Confidence</div>
            <div class="value green">{avg:.0%}</div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_panel(state):
    with state.lock:
        label = state.current_label
        conf = state.current_confidence
        status = state.current_status
    if label and status in ("sign_detected", "recognizing", "low_confidence"):
        st.markdown(f"""
        <div class="prediction-badge">
            <div class="sign-label">{label.upper()}</div>
            <div class="confidence">{conf:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-prediction">
            No prediction yet.<br/>Start the camera and show a hand sign.
        </div>
        """, unsafe_allow_html=True)


def render_history_list(history):
    if not history:
        st.markdown("""
        <div style="color:rgba(255,255,255,0.4); font-size:0.85rem; text-align:center; padding:1rem 0;">
            Recognition history will appear here as signs are accepted.
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <div class="history-header">
        <span style="width:25%">Time</span>
        <span style="width:50%">Sign</span>
        <span style="width:25%; text-align:right">Confidence</span>
    </div>
    """, unsafe_allow_html=True)

    rows = ""
    for entry in reversed(history[-25:]):
        rows += f"""
        <div class="history-row">
            <span class="time">{entry['time']}</span>
            <span class="sign">{entry['label'].upper()}</span>
            <span class="conf">{entry['confidence']:.0%}</span>
        </div>
        """
    st.markdown(rows, unsafe_allow_html=True)


def render_instructions(model, label_encoder):
    classes = label_encoder.classes_ if label_encoder is not None else model.classes_
    alpha = sorted([c for c in classes if len(c) == 1 and c.isalpha()])
    digits = sorted([c for c in classes if c.isdigit()])
    words = sorted([c for c in classes if len(c) > 1 or (len(c) == 1 and not c.isalpha())])
    parts = []
    if alpha:
        parts.append(f"<strong>Alphabet:</strong> {' '.join(alpha)}")
    if digits:
        parts.append(f"<strong>Numbers:</strong> {' '.join(digits)}")
    if words:
        parts.append(f"<strong>Words:</strong> {', '.join(words)}")
    sign_html = " &nbsp;|&nbsp; ".join(parts)

    st.markdown(f"""
    <div class="instruction-callout">
        <div class="icon">💡</div>
        <div>
            <strong>How to use:</strong> Start the camera, position your hand clearly in the frame,
            and hold your sign steady. The model recognizes <strong>{len(classes)} signs</strong>.
            A prediction is accepted only when it remains stable across multiple consecutive frames
            with high confidence.<br/><br/>
            {sign_html}<br/><br/>
            <strong>Tip:</strong> If predictions feel too sensitive, raise the Confidence Threshold
            or Stability Frames in the sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    render_brand_header()

    # ── Load resources ────────────────────────────────────
    try:
        model, scaler, label_encoder = load_ml_model()
    except Exception as e:
        st.error(f"Failed to load the trained model: {e}")
        st.info("Run `python train_model.py` to train the model first.")
        st.stop()

    if model is None:
        st.error("Model files not found. Please train the model first.")
        st.info("Run locally: `python train_model.py`")
        st.stop()

    try:
        landmarker = load_hand_landmarker()
    except Exception as e:
        st.error(f"Failed to load hand landmarker: {e}")
        st.info("Make sure `hand_landmarker.task` is in the project directory.")
        st.stop()

    # ── Session state ─────────────────────────────────────
    if "recog_state" not in st.session_state:
        st.session_state.recog_state = RecognitionState()
    state: RecognitionState = st.session_state.recog_state

    # ── Sidebar (updates state config live) ───────────────
    render_sidebar(model, label_encoder, state)

    # ── Tabs: Live Camera / Photo Capture ──────────────────
    tab_live, tab_photo = st.tabs(["📹 Live Camera", "📷 Photo Capture"])

    # ── LIVE CAMERA TAB ────────────────────────────────────
    with tab_live:
        # Track if camera is in use (for warning in Photo Capture tab)
        if "camera_in_use" not in st.session_state:
            st.session_state.camera_in_use = False
        if "camera_start_requested" not in st.session_state:
            st.session_state.camera_start_requested = False

        col_feed, col_panel = st.columns([3, 2], gap="medium")

        with col_feed:
            st.markdown("#### Camera Preview")

            # Build callback (captures current state + resources)
            callback = make_video_frame_callback(state, landmarker, model, scaler, label_encoder)

            webrtc_ctx = None

            if st.session_state.camera_start_requested:
                # Only render webrtc_streamer when the user explicitly
                # clicked Start.  This prevents the browser from auto-
                # grabbing getUserMedia on every page load, which is
                # the root cause of NotReadableError: Device in use
                # when a stale stream from a previous session still
                # holds the device handle.
                try:
                    webrtc_ctx = webrtc_streamer(
                        key="signcam",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration={
                            "iceServers": []  # No STUN needed for localhost
                        },
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": 640},
                                "height": {"ideal": 480},
                                "facingMode": "user"
                            },
                            "audio": False
                        },
                        video_frame_callback=callback,
                        async_processing=True,
                        desired_playing_state=True,
                    )
                except Exception as e:
                    err_msg = str(e).lower()
                    st.session_state.camera_start_requested = False
                    st.session_state.camera_start_error = (
                        "no_device" if "notfound" in err_msg
                        else "device_busy" if "notreadable" in err_msg or "in use" in err_msg
                        else "other"
                    )
                    webrtc_ctx = None

                # Sync flags with the actual webrtc state
                if webrtc_ctx and webrtc_ctx.state.playing:
                    st.session_state.camera_in_use = True
                    with state.lock:
                        if state.current_status in ("stopped", "ready"):
                            state.current_status = "ready"
                elif webrtc_ctx and not webrtc_ctx.state.playing and not webrtc_ctx.state.signalling:
                    st.session_state.camera_in_use = False
                    with state.lock:
                        if state.current_status not in ("ready", "sign_detected"):
                            state.current_status = "stopped" if state.history else "ready"
            else:
                # Show a clean placeholder until the user clicks Start
                err = st.session_state.get("camera_start_error")
                st.markdown("""
                <div class="camera-start-placeholder">
                    <div class="icon">📹</div>
                    <div class="title">Camera Ready</div>
                    <div class="hint">Click <strong>Start Camera</strong> below to begin live recognition</div>
                </div>
                """, unsafe_allow_html=True)

                if err == "device_busy":
                    st.error(
                        "Camera is busy — another app or tab is using it. "
                        "Close other camera apps, then click **Release Camera** in the sidebar and try again."
                    )
                elif err == "no_device":
                    st.error(
                        "No camera detected. Make sure a webcam is connected and enabled."
                    )
                elif err == "other":
                    st.error(
                        "Camera could not start. Try refreshing the page or using a different browser."
                    )

                if st.button("Start Camera", use_container_width=True, type="primary"):
                    st.session_state.camera_start_requested = True
                    st.session_state.camera_start_error = None
                    st.rerun()

            # Troubleshooting (collapsible — does not steal viewport space)
            with st.expander("Camera not starting? Troubleshooting tips"):
                st.markdown("""
                | Error | Cause | Fix |
                |-------|-------|-----|
                | `NotReadableError: Device in use` | Another app/tab has the camera | Close other apps or click **Release Camera** in sidebar |
                | `NotAllowedError` | Camera permission denied | Click the camera icon in the address bar and allow access |
                | `NotFoundError` | No camera detected | Check that your webcam is connected and enabled |
                | Camera won't start | Not a secure context | Use `http://localhost` or `https://` |
                | Black screen | Stream not initializing | Refresh the page (Ctrl+F5) |

                **Quick fixes:**
                1. Close other apps using the camera (Zoom, Teams, other tabs)
                2. Click **Release Camera** in the sidebar
                3. Refresh the page (Ctrl+F5)
                4. Check browser permissions — lock/camera icon in the address bar
                """)

            # Live autorefresh only while the stream is active
            if webrtc_ctx and (webrtc_ctx.state.playing or webrtc_ctx.state.signalling):
                st_autorefresh(interval=600, key="live_ui_refresh")

            render_instructions(model, label_encoder)

        with col_panel:
            st.markdown('<div class="panel-tight">', unsafe_allow_html=True)

            st.markdown("#### Recognition Status")
            render_status_panel(state)

            st.markdown("#### Current Prediction")
            render_prediction_panel(state)

            st.markdown("#### Session Statistics")
            with state.lock:
                history_snapshot = list(state.history)
            render_stat_cards(history_snapshot)

            st.markdown("#### Recognition History")
            render_history_list(history_snapshot)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── PHOTO CAPTURE TAB (fallback / single-shot) ────────
    with tab_photo:
        st.markdown("#### Single Photo Capture")

        # Show warning if Live Camera is active
        if st.session_state.get("camera_in_use", False):
            st.warning(
                "⚠️ **The Live Camera is currently active.** "
                "The camera can only be used by one mode at a time. "
                "Please **click the 'Stop' button** in the Live Camera tab before using Photo Capture."
            )
        else:
            st.info(
                "This mode takes a single photo and predicts the sign. "
                "For continuous recognition with stability filtering, use the **Live Camera** tab."
            )

        camera_image = st.camera_input(
            "Take a photo of your hand sign",
            help="Position your hand clearly in the frame and click the capture button",
        )

        if camera_image is not None:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(mp_image)

                hand_detected = len(results.hand_landmarks) > 0
                if hand_detected:
                    _draw_hand_skeleton(frame, results.hand_landmarks[0])
                    hand_lms = results.hand_landmarks[0]
                    feats = np.array(
                        [c for lm in hand_lms for c in (lm.x, lm.y, lm.z)],
                        dtype=np.float64,
                    ).reshape(1, -1)
                    feats_scaled = scaler.transform(feats)
                    probs = model.predict_proba(feats_scaled)[0]
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])
                    enc_class = model.classes_[idx]
                    label = label_encoder.inverse_transform([enc_class])[0] if label_encoder else str(enc_class)

                    if conf >= state.threshold:
                        _draw_overlay(frame, f"{label.upper()}  {conf:.0%}", (0, 255, 100))
                        with state.lock:
                            if not state.history or state.history[-1]["label"] != label:
                                state.history.append({
                                    "label": label,
                                    "confidence": conf,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                })
                    else:
                        _draw_overlay(frame, f"Low conf: {conf:.0%}", (100, 180, 255))
                else:
                    _draw_overlay(frame, "No hand detected", (180, 180, 180))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb)
            else:
                st.error("Could not decode the captured image.")
        else:
            st.markdown("""
            <div class="camera-idle">
                <div class="icon">📷</div>
                <div style="font-weight:500; color:rgba(255,255,255,0.6);">Camera ready</div>
                <div style="margin-top:0.3rem;">Click the camera button above to capture a sign.</div>
            </div>
            """, unsafe_allow_html=True)

        # Camera troubleshooting
        st.markdown("")
        with st.expander("🔧 Camera not working? Click here for help"):
            st.markdown("""
            **Common camera issues and solutions:**

            | Error | Cause | Solution |
            |-------|-------|----------|
            | `NotReadableError: Device in use` | Another app or browser tab is using the camera | Close other apps/tabs using the camera, or **click 'Stop'** in the Live Camera tab |
            | `NotAllowedError` | Camera permission denied | Click the camera icon in your browser's address bar and allow camera access |
            | `NotFoundError` | No camera detected | Check that your webcam is connected and not disabled |
            | Camera won't start | Accessing via HTTP (not localhost) | Use `http://localhost:8501` or `https://` (browsers require secure context for camera) |
            | Black screen | Camera permissions granted but stream not starting | Refresh the page and try again |

            **Quick fixes:**
            1. **Close other apps** that might be using the camera (Zoom, Teams, other browser tabs)
            2. **Refresh the page** (Ctrl+F5 or Cmd+Shift+R)
            3. **Check browser permissions** — click the lock/camera icon in the address bar
            4. **Try a different browser** (Chrome, Firefox, Edge all work)
            5. **Restart the app** if the camera is stuck
            """)


if __name__ == "__main__":
    main()
