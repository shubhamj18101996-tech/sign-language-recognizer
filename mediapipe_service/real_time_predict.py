# real_time_predict.py
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from collections import deque
from model import LSTMClassifier
import requests

# Configuration
MODEL_DIR = Path("data/models")
model_path = MODEL_DIR / "model.pt"
labels_path = MODEL_DIR / "labels_map.json"
EXTRACTOR_URL = "http://127.0.0.1:8000/extract/image"
SEQ_LEN = 40
CONFIDENCE_THRESHOLD = 0.30
FRAME_SKIP = 3  # Process every 2nd frame (reduces lag)

# Load model and labels
print("Loading model...")
with open(labels_path) as f:
    idx_to_label = json.load(f)

num_classes = len(idx_to_label)
model = LSTMClassifier(input_dim=258, hidden_dim=128, num_layers=2, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print(f"✓ Model loaded")
print(f"Labels: {list(idx_to_label.values())}")
print("\n" + "="*60)
print("Starting live gesture recognition (press 'q' to quit)")
print("="*60 + "\n")

# Capture landmarks from live video
landmark_buffer = deque(maxlen=SEQ_LEN)
frame_count = 0
prediction_count = 0

cap = cv2.VideoCapture(0)  # 0 = default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_count += 1

    # Skip some frames to reduce lag - display all, process only every Nth
    if frame_count % FRAME_SKIP != 0:
        cv2.putText(frame, f"Frames: {len(landmark_buffer)}/{SEQ_LEN} [skipped]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        cv2.imshow("Live Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Encode frame as JPEG bytes (only for non-skipped frames)
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not success:
        continue

    # Send to extractor service
    try:
        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
        response = requests.post(EXTRACTOR_URL, files=files, timeout=3)

        if response.status_code == 200:
            data = response.json()
            landmarks = data.get('landmarks', [])

            if landmarks:
                landmark_buffer.append(landmarks)

                # Once we have enough frames, predict
                if len(landmark_buffer) == SEQ_LEN:
                    seq = np.array(list(landmark_buffer), dtype=np.float32)
                    x = torch.tensor(seq[None, ...], dtype=torch.float32)

                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=-1).numpy()[0]
                        pred_idx = int(probs.argmax())
                        confidence = float(probs.max())
                        pred_label = idx_to_label[str(pred_idx)]

                    # Print prediction if confidence is high
                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction_count += 1
                        print(f"[Frame {frame_count:4d}] 🎯 GESTURE: {pred_label.upper():12s} | Confidence: {confidence:.2%}")
                    else:
                        print(f"[Frame {frame_count:4d}] ❓ Uncertain: {pred_label} ({confidence:.2%})")

    except requests.exceptions.Timeout:
        print(f"[Frame {frame_count:4d}] ⚠️  Timeout (extractor slow)")
    except Exception as e:
        print(f"[Frame {frame_count:4d}] ❌ Error: {e}")

    # Display frame with info
    cv2.putText(frame, f"Frames: {len(landmark_buffer)}/{SEQ_LEN}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Live Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print(f"Session complete: {frame_count} frames, {prediction_count} predictions")
print("="*60)