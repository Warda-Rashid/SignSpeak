"""
generate_demo_data.py
---------------------
Generates synthetic hand landmark data for all 85 signs so you can
train and test the model immediately.  Each sign gets a unique but
deterministic landmark pattern with random variation, giving the
classifier something to learn.

After training with this data the app will work, but accuracy on
real hand gestures will be limited.  Collect real samples with
data_collection.py later to improve accuracy.

Usage:
    python generate_demo_data.py
    python train_model.py
"""

import csv
import os
import numpy as np
import sys

# Import sign labels from data_collection
sys.path.insert(0, os.path.dirname(__file__))
from data_collection import SIGN_LABELS, SIGN_CATEGORIES, TARGET_SAMPLES

DATA_FILE = "sign_data.csv"

def generate_landmark_pattern(label, seed):
    """
    Generate a deterministic 63-feature landmark vector (21 landmarks x 3 coords)
    that is unique per label with clearly distinct patterns per class.
    """
    rng = np.random.RandomState(seed)

    base = np.zeros((21, 3))

    # Create a strong unique signature from the label
    # Each label gets its own deterministic random state for position generation
    label_rng = np.random.RandomState(hash(label) % (2**31))
    label_pos = label_rng.uniform(0.1, 0.9, size=(21, 3))

    # Finger base positions (spread across the hand)
    finger_bases = {
        'thumb':  np.array([0.35, 0.65, 0.0]),
        'index':  np.array([0.40, 0.45, 0.0]),
        'middle': np.array([0.50, 0.40, 0.0]),
        'ring':   np.array([0.58, 0.42, 0.0]),
        'pinky':  np.array([0.65, 0.48, 0.0]),
    }

    # Wrist
    wrist_offset = label_rng.uniform(-0.1, 0.1, size=3)
    base[0] = np.array([0.5, 0.85, 0.0]) + wrist_offset

    # Each finger has 4 joints; use label-specific angles to curl/extend
    finger_groups = [
        (1, 5, 'thumb'),
        (5, 9, 'index'),
        (9, 13, 'middle'),
        (13, 17, 'ring'),
        (17, 21, 'pinky'),
    ]

    for start, end, finger_name in finger_groups:
        fb = finger_bases[finger_name]
        # Per-label curl amount and spread direction
        curl = label_rng.uniform(0.0, 0.35)       # how curled the finger is
        spread = label_rng.uniform(-0.15, 0.15)    # lateral spread
        z_twist = label_rng.uniform(-0.1, 0.1)     # z-axis twist
        length = label_rng.uniform(0.04, 0.08)     # segment length

        for i, lm_idx in enumerate(range(start, end)):
            # Each joint extends from previous, with label-specific curl
            if lm_idx == start:
                prev = base[0].copy()
            else:
                prev = base[lm_idx - 1].copy()

            # Direction: mostly upward with finger-specific curl
            dy = -length + curl * 0.03 * (i + 1)
            dx = spread * 0.5 + label_rng.uniform(-0.02, 0.02)
            dz = z_twist * 0.3 + label_rng.uniform(-0.01, 0.01)

            base[lm_idx] = prev + np.array([dx, dy, dz])

    # Add label-specific global transform (rotation + shift)
    shift = label_rng.uniform(-0.08, 0.08, size=3)
    base += shift

    # Clip to valid range
    base = np.clip(base, 0.01, 0.99)

    return base.flatten()


def main():
    print("=" * 65)
    print("  SignSpeak - Demo Data Generator")
    print("=" * 65)
    print(f"  Signs:    {len(SIGN_LABELS)}")
    print(f"  Samples per sign: {TARGET_SAMPLES}")
    print(f"  Total samples:    {len(SIGN_LABELS) * TARGET_SAMPLES}")
    print(f"  Output:   {DATA_FILE}")
    print("=" * 65)

    # Warn if file already exists
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            existing = sum(1 for _ in f) - 1  # subtract header
        if existing > 0:
            print(f"\n  Existing data found: {existing} samples in {DATA_FILE}")
            print("  New synthetic data will be APPENDED to existing data.")
            response = input("  Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("  Aborted.")
                return

    # Write CSV
    file_exists = os.path.exists(DATA_FILE)
    samples_written = 0

    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)

        # Header
        if not file_exists:
            header = [f"x{i}" for i in range(21)] + \
                     [f"y{i}" for i in range(21)] + \
                     [f"z{i}" for i in range(21)] + \
                     ["label"]
            writer.writerow(header)

        for label_idx, label in enumerate(SIGN_LABELS):
            # Unique seed per label
            base_seed = label_idx * 1000

            for sample_idx in range(TARGET_SAMPLES):
                seed = base_seed + sample_idx
                landmarks = generate_landmark_pattern(label, seed)

                # Add natural noise
                noise = np.random.RandomState(seed + 9999).normal(0, 0.006, landmarks.shape)
                landmarks = landmarks + noise

                # Clip to valid range [0, 1]
                landmarks = np.clip(landmarks, 0.0, 1.0)

                row = landmarks.tolist() + [label]
                writer.writerow(row)
                samples_written += 1

            # Progress indicator
            progress = (label_idx + 1) / len(SIGN_LABELS) * 100
            bar_len = 30
            filled = int(bar_len * (label_idx + 1) / len(SIGN_LABELS))
            bar = "#" * filled + "." * (bar_len - filled)
            print(f"\r  [{bar}] {progress:5.1f}%  {label:20s} ({samples_written} samples)", end="", flush=True)

    print(f"\n\n  Done! {samples_written} synthetic samples written to {DATA_FILE}")
    print(f"\n  Next step: python train_model.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
