\
import cv2
import sys

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    sys.exit(1)

print(f"Camera opened successfully")
ret, frame = cap.read()

if ret:
    print(f"Frame captured: shape={frame.shape}")
else:
    print("ERROR: Could not read frame")

cap.release()
print("Camera released")
