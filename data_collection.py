"""
data_collection.py
------------------
Collects sign language samples using MediaPipe Hands.
Opens webcam, detects hand landmarks, and saves labeled samples to CSV.
Supports full ASL alphabet (A-Z), numbers (0-9), and common phrases.

Usage:
    python data_collection.py

Controls:
    SPACE  - Capture a sample for the current label
    n      - Move to next label
    p      - Move to previous label
    c      - Jump to next category
    q      - Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import os
import numpy as np
from collections import OrderedDict

# Hand landmark connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)             # Palm
]

# ─────────────────────────────────────────────────────────
# Full sign vocabulary organized by category
# ─────────────────────────────────────────────────────────
SIGN_CATEGORIES = OrderedDict([
    ("Alphabet", [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    ]),
    ("Numbers", [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]),
    ("Greetings", [
        "hello", "goodbye", "good_morning", "good_night",
        "nice_to_meet_you", "how_are_you"
    ]),
    ("Common Words", [
        "yes", "no", "please", "thank_you", "sorry",
        "help", "stop", "name", "good", "bad",
        "more", "done", "want", "need", "like",
        "love", "friend", "family", "home", "school",
        "work", "eat", "drink", "sleep", "go",
        "come", "see", "hear", "understand", "think"
    ]),
    ("Questions", [
        "what", "where", "when", "who", "why", "how"
    ]),
    ("Time", [
        "today", "tomorrow", "yesterday", "morning",
        "night", "now", "later"
    ]),
])

# Flatten all labels for iteration
SIGN_LABELS = []
LABEL_TO_CATEGORY = {}
CATEGORY_RANGES = {}

idx = 0
for category, labels in SIGN_CATEGORIES.items():
    start = idx
    for label in labels:
        SIGN_LABELS.append(label)
        LABEL_TO_CATEGORY[label] = category
        idx += 1
    CATEGORY_RANGES[category] = (start, idx)

# CSV file to store collected data
DATA_FILE = "sign_data.csv"

# Target samples per label
TARGET_SAMPLES = 50

def draw_text_with_bg(frame, text, position, font_scale, color, thickness=2, bg_color=(0, 0, 0)):
    """Draw text with a semi-transparent background for readability."""
    x, y = position
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    padding = 6
    # Draw background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_progress_bar(frame, x, y, width, height, progress, color=(0, 200, 100)):
    """Draw a simple progress bar on the frame."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Fill
    fill_w = int(width * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (120, 120, 120), 1)

def main():
    # Initialize MediaPipe Hand Landmarker
    base_options = mp_python.BaseOptions(
        model_asset_path='hand_landmarker.task'
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    total_labels = len(SIGN_LABELS)
    categories = list(SIGN_CATEGORIES.keys())

    # Count existing samples if CSV already exists
    sample_count = {label: 0 for label in SIGN_LABELS}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) > 0 and row[-1] in sample_count:
                    sample_count[row[-1]] += 1

    print("=" * 65)
    print("  SignSpeak Data Collection - Full Vocabulary")
    print("=" * 65)
    print(f"  Total signs:  {total_labels}")
    print(f"  Categories:   {len(categories)}")
    for cat, labels in SIGN_CATEGORIES.items():
        print(f"    {cat:20s}  ({len(labels)} signs)")
    print("-" * 65)
    print("  Controls:")
    print("    SPACE  - Capture sample for current label")
    print("    n / p  - Next / Previous label")
    print("    c      - Jump to next category")
    print("    q      - Quit")
    print("-" * 65)
    print(f"  Target: {TARGET_SAMPLES} samples per sign")
    print(f"  Data file: {DATA_FILE}")
    print("=" * 65)

    # Prepare CSV file
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 21 landmarks * 3 coordinates (x, y, z) = 63 features + 1 label
            header = [f"x{i}" for i in range(21)] + \
                     [f"y{i}" for i in range(21)] + \
                     [f"z{i}" for i in range(21)] + \
                     ["label"]
            writer.writerow(header)

    current_label_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Draw hand landmarks
        hand_detected = False
        if results.hand_landmarks:
            hand_detected = True
            h, w, _ = frame.shape
            for hand_landmarks in results.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (0, 200, 120), 2)
                for pt in points:
                    cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                    cv2.circle(frame, pt, 5, (255, 255, 255), 1)

        # ── UI Overlay ──────────────────────────────────────
        current_label = SIGN_LABELS[current_label_idx]
        current_category = LABEL_TO_CATEGORY[current_label]
        current_count = sample_count[current_label]
        progress = current_count / TARGET_SAMPLES

        # Top-left: Category and label
        draw_text_with_bg(frame, f"[{current_category}]", (12, 28), 0.55, (180, 180, 255), 1)
        draw_text_with_bg(frame, f"Sign: {current_label}", (12, 58), 1.0, (0, 255, 100), 2)
        draw_text_with_bg(frame, f"Samples: {current_count} / {TARGET_SAMPLES}", (12, 88), 0.6, (255, 255, 255), 1)

        # Progress bar for current label
        draw_progress_bar(frame, 12, 98, 200, 10, progress,
                          color=(0, 200, 100) if progress >= 1.0 else (0, 160, 255))

        # Top-right: Overall progress
        total_collected = sum(sample_count.values())
        total_needed = total_labels * TARGET_SAMPLES
        overall_progress = total_collected / total_needed if total_needed > 0 else 0
        draw_text_with_bg(frame, f"{current_label_idx + 1} / {total_labels}",
                          (frame.shape[1] - 120, 28), 0.6, (255, 255, 255), 1)
        draw_text_with_bg(frame, f"Total: {total_collected}/{total_needed}",
                          (frame.shape[1] - 160, 55), 0.5, (200, 200, 200), 1)
        draw_progress_bar(frame, frame.shape[1] - 170, 65, 160, 8, overall_progress)

        # Bottom: Controls
        draw_text_with_bg(frame, "SPACE=capture  n=next  p=prev  c=category  q=quit",
                          (12, frame.shape[0] - 15), 0.45, (200, 200, 200), 1)

        # Hand detection indicator
        if hand_detected:
            cv2.circle(frame, (frame.shape[1] - 25, frame.shape[0] - 25), 10, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (frame.shape[1] - 25, frame.shape[0] - 25), 10, (0, 0, 200), -1)

        # Flash green border when sample captured
        if current_count >= TARGET_SAMPLES:
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 200, 100), 3)

        cv2.imshow("SignSpeak Data Collection", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting data collection...")
            break
        elif key == ord('n'):
            current_label_idx = (current_label_idx + 1) % total_labels
            new_label = SIGN_LABELS[current_label_idx]
            print(f"  -> Next: {new_label} [{LABEL_TO_CATEGORY[new_label]}]")
        elif key == ord('p'):
            current_label_idx = (current_label_idx - 1) % total_labels
            new_label = SIGN_LABELS[current_label_idx]
            print(f"  <- Prev: {new_label} [{LABEL_TO_CATEGORY[new_label]}]")
        elif key == ord('c'):
            # Jump to the start of the next category
            cat_start = None
            for cat, (start, end) in CATEGORY_RANGES.items():
                if start > current_label_idx:
                    cat_start = start
                    break
            if cat_start is None:
                cat_start = 0  # wrap around
            current_label_idx = cat_start
            new_label = SIGN_LABELS[current_label_idx]
            print(f"  >> Category jump: {LABEL_TO_CATEGORY[new_label]} -> {new_label}")
        elif key == ord(' '):
            if hand_detected:
                hand_landmarks = results.hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks.append(current_label)

                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)

                sample_count[current_label] += 1
                new_count = sample_count[current_label]

                if new_count == TARGET_SAMPLES:
                    print(f"  [COMPLETE] '{current_label}' reached {TARGET_SAMPLES} samples!")
                elif new_count % 10 == 0:
                    print(f"  Captured #{new_count} for '{current_label}'")
            else:
                print("  [!] No hand detected - show your hand to the camera.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Summary
    print("\n" + "=" * 65)
    print("  Data Collection Complete!")
    print("=" * 65)

    total_collected = sum(sample_count.values())
    print(f"\n  Total samples: {total_collected}")

    for category, labels in SIGN_CATEGORIES.items():
        cat_total = sum(sample_count[l] for l in labels)
        cat_target = len(labels) * TARGET_SAMPLES
        status = "OK" if cat_total >= cat_target else f"{cat_target - cat_total} more needed"
        print(f"\n  [{category}] {cat_total}/{cat_target} - {status}")
        for label in labels:
            count = sample_count[label]
            bar = "#" * min(count, 40) + "." * max(0, min(40, TARGET_SAMPLES - count))
            mark = "v" if count >= TARGET_SAMPLES else " "
            print(f"    {label:12s} [{bar}] {count:3d} {mark}")

    print(f"\n  Data saved to: {DATA_FILE}")
    print(f"  Next step: python train_model.py")
    print("=" * 65)

if __name__ == "__main__":
    main()
