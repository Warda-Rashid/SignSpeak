"""
train_model.py
--------------
Trains a machine learning model on collected sign language data.
Loads CSV, splits into train/test, trains KNN + MLP + RandomForest,
selects best model, and saves it.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from collections import Counter

# Configuration
DATA_FILE = "sign_data.csv"
MODEL_FILE = "sign_model.pkl"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 2  # minimum to include in training

def load_data():
    """Load and prepare the dataset from CSV."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        print("Please run data_collection.py first to collect samples.")
        return None, None, None
    
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y_raw = df['label'].values
    
    print(f"Loaded {len(X)} total samples")
    
    # Show per-class counts
    counts = Counter(y_raw)
    labels_sorted = sorted(counts.keys())
    
    # Detect categories based on label patterns
    alphabet_labels = sorted([l for l in labels_sorted if len(l) == 1 and l.isalpha()])
    number_labels = sorted([l for l in labels_sorted if l.isdigit()])
    word_labels = sorted([l for l in labels_sorted if len(l) > 1 or (len(l) == 1 and not l.isalpha())])
    
    print(f"\n  Alphabet ({len(alphabet_labels)} letters): ", end="")
    print(", ".join(alphabet_labels) if alphabet_labels else "none")
    print(f"  Numbers  ({len(number_labels)} digits):  ", end="")
    print(", ".join(number_labels) if number_labels else "none")
    print(f"  Words    ({len(word_labels)} signs):   ", end="")
    print(", ".join(word_labels) if word_labels else "none")
    
    # Filter out classes with too few samples (need at least MIN_SAMPLES_PER_CLASS)
    valid_labels = [l for l, c in counts.items() if c >= MIN_SAMPLES_PER_CLASS]
    filtered_mask = np.isin(y_raw, valid_labels)
    X = X[filtered_mask]
    y_raw = y_raw[filtered_mask]
    
    skipped = [l for l, c in counts.items() if c < MIN_SAMPLES_PER_CLASS]
    if skipped:
        print(f"\n  Skipped {len(skipped)} classes with < {MIN_SAMPLES_PER_CLASS} samples: {skipped}")
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print(f"\n  Training with {len(np.unique(y))} classes, {len(X)} samples")
    print(f"  Samples per class:")
    for label in label_encoder.classes_:
        count = np.sum(y_raw == label)
        bar = "#" * min(count, 30)
        print(f"    {label:20s}  {count:4d}  {bar}")
    
    return X, y, label_encoder

def train_knn(X_train, y_train, X_test, y_test):
    """Train KNN classifier and return model and accuracy."""
    print("\n" + "=" * 60)
    print("  Training KNN Classifier...")
    print("=" * 60)
    
    best_k = 3
    best_accuracy = 0
    
    # Scale k search based on dataset size
    max_k = min(21, len(X_train) // 2)
    
    for k in range(1, max_k + 1, 2):  # odd values only to avoid ties
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  k={k:2d}: accuracy = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    print(f"\n  Best k = {best_k} with accuracy = {best_accuracy:.4f}")
    
    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn.fit(X_train, y_train)
    
    return knn, best_accuracy

def train_mlp(X_train, y_train, X_test, y_test):
    """Train MLP (Neural Network) classifier."""
    print("\n" + "=" * 60)
    print("  Training MLP Classifier (Neural Network)...")
    print("=" * 60)
    
    n_classes = len(np.unique(y_train))
    
    # Scale network size based on data
    if n_classes > 30:
        hidden = (256, 128, 64)
        max_iter = 800
    elif n_classes > 15:
        hidden = (128, 64, 32)
        max_iter = 600
    else:
        hidden = (64, 32)
        max_iter = 500
    
    print(f"  Architecture: {hidden}")
    print(f"  Max iterations: {max_iter}")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=min(64, len(X_train) // 4),
        n_iter_no_change=20,
        activation='relu'
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  MLP accuracy: {accuracy:.4f}")
    print(f"  Training iterations: {mlp.n_iter_}")
    
    return mlp, accuracy

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train RandomForest classifier."""
    print("\n" + "=" * 60)
    print("  Training RandomForest Classifier...")
    print("=" * 60)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  RandomForest accuracy: {accuracy:.4f}")
    
    # Show top feature importances
    top_n = 5
    top_indices = np.argsort(rf.feature_importances_)[-top_n:]
    print(f"  Top {top_n} important features:")
    for idx in reversed(top_indices):
        print(f"    Feature {idx}: importance = {rf.feature_importances_[idx]:.4f}")
    
    return rf, accuracy

def main():
    print("=" * 65)
    print("  SignSpeak Model Training - Full Vocabulary")
    print("=" * 65)
    
    # Load data
    X, y, label_encoder = load_data()
    if X is None:
        return
    
    n_classes = len(np.unique(y))
    
    # Check if we have enough data
    if len(X) < n_classes * 2:
        print("\n  Warning: Very few samples per class!")
        print("  Recommended: at least 20-30 samples per sign")
        print("  Collect more data for better accuracy.")
    
    # Split into train and test sets
    # Handle case where some classes have very few samples
    min_class_count = min(np.bincount(y))
    
    if min_class_count < 2:
        # Can't do stratified split with 1 sample per class
        print("\n  Some classes have only 1 sample - using non-stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    
    print(f"\n  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train all three models
    results = {}
    
    knn_model, knn_accuracy = train_knn(X_train_scaled, y_train, X_test_scaled, y_test)
    results['KNN'] = (knn_model, knn_accuracy)
    
    rf_model, rf_accuracy = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    results['RandomForest'] = (rf_model, rf_accuracy)
    
    mlp_model, mlp_accuracy = train_mlp(X_train_scaled, y_train, X_test_scaled, y_test)
    results['MLP'] = (mlp_model, mlp_accuracy)
    
    # Choose the best model
    print("\n" + "=" * 65)
    print("  Model Comparison")
    print("=" * 65)
    
    for name, (model, acc) in results.items():
        bar = "#" * int(acc * 40)
        print(f"  {name:15s}  {acc:.4f}  ({acc*100:.1f}%)  {bar}")
    
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_accuracy = results[best_name]
    
    print(f"\n  Winner: {best_name} with {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
    
    # Print detailed classification report for best model
    print("\n" + "=" * 65)
    print(f"  Classification Report ({best_name})")
    print("=" * 65)
    y_pred = best_model.predict(X_test_scaled)
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Save model, scaler, and label encoder
    print("=" * 65)
    print("  Saving model...")
    print("=" * 65)
    
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    
    print(f"  Model:         {MODEL_FILE}")
    print(f"  Scaler:        {SCALER_FILE}")
    print(f"  Label encoder: {LABEL_ENCODER_FILE}")
    print(f"  Model type:    {best_name}")
    print(f"  Classes:       {n_classes}")
    print(f"  Test accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print("=" * 65)
    print("\n  Next step: streamlit run app.py")

if __name__ == "__main__":
    main()
