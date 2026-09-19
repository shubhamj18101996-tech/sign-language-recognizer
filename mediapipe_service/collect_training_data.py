# collect_training_data.py
import cv2
import numpy as np
from pathlib import Path
import requests
import time

DATA_DIR = Path("data/landmarks")
DATA_DIR.mkdir(parents=True, exist_ok=True)

GESTURES = ["yes", "thankyou", "hello", "help", "no", "iloveyou"]
EXTRACTOR_URL = "http://127.0.0.1:8000/extract/image"
SAMPLES_PER_GESTURE = 10
FRAMES_PER_SAMPLE = 40

print("="*60)
print("SIGN LANGUAGE TRAINING DATA COLLECTOR")
print("="*60)
print(f"\nGestures to collect: {', '.join(GESTURES)}")
print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
print(f"Frames per sample: {FRAMES_PER_SAMPLE}")
print("\nControls:")
print("  SPACE = Start recording current sample")
print("  Q     = Skip current gesture")
print("  ESC   = Exit completely")
print("\n" + "="*60 + "\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

total_samples_collected = 0

for gesture_idx, gesture in enumerate(GESTURES):
    print(f"\n[{gesture_idx+1}/{len(GESTURES)}] Collecting: {gesture.upper()}")
    print("-" * 60)

    sample_count = 0
    skip_gesture = False

    while sample_count < SAMPLES_PER_GESTURE and not skip_gesture:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Display instructions
        cv2.putText(frame, f"Gesture: {gesture.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Sample {sample_count+1}/{SAMPLES_PER_GESTURE}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=Record  Q=Skip  ESC=Quit",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Collect Training Data - Press SPACE to record", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("⚠️  Exiting...")
            skip_gesture = True
            break
        elif key == ord('q'):  # Q
            print(f"⏭️  Skipping {gesture}")
            skip_gesture = True
            break
        elif key == ord(' '):  # SPACE
            print(f"  Recording sample {sample_count+1}/{SAMPLES_PER_GESTURE}...", end='', flush=True)
            landmarks_sequence = []

            for frame_idx in range(FRAMES_PER_SAMPLE):
                ret, frame = cap.read()
                if not ret:
                    print(" ❌ Failed to capture")
                    break

                frame = cv2.flip(frame, 1)

                # Show recording progress
                cv2.putText(frame, f"RECORDING: {frame_idx+1}/{FRAMES_PER_SAMPLE}",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (10, 60), (10 + frame_idx * 5, 80), (0, 255, 0), -1)
                cv2.imshow("Collect Training Data - Press SPACE to record", frame)
                cv2.waitKey(30)

                # Extract landmarks
                success, buffer = cv2.imencode('.jpg', frame)
                if success:
                    try:
                        files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                        response = requests.post(EXTRACTOR_URL, files=files, timeout=3)

                        if response.status_code == 200:
                            data = response.json()
                            landmarks = data.get('landmarks', [])
                            if landmarks:
                                landmarks_sequence.append(landmarks)
                    except Exception as e:
                        pass

            # Save if we got enough frames
            if len(landmarks_sequence) == FRAMES_PER_SAMPLE:
                sequence_array = np.array(landmarks_sequence, dtype=np.float32)
                output_path = DATA_DIR / f"{gesture}_{sample_count+1:02d}.npy"
                np.save(output_path, sequence_array)
                print(f" ✓ Saved")
                sample_count += 1
                total_samples_collected += 1
                time.sleep(0.5)  # Brief pause between samples
            else:
                print(f" ❌ Incomplete ({len(landmarks_sequence)}/{FRAMES_PER_SAMPLE} frames)")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print(f"✓ Data collection complete!")
print(f"  Total samples collected: {total_samples_collected}")
print(f"  Location: {DATA_DIR}")
print("="*60)
print("\nNext steps:")
print("  1. Run: python train.py")
print("  2. Run: python real_time_predict.py")
print("="*60)