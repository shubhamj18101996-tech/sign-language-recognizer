# generate_synthetic_data.py
import numpy as np
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("data/landmarks")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define signs and generate synthetic data
SIGNS = ["hello", "thankyou", "yes", "no", "help", "iloveyou"]
NUM_CLIPS_PER_SIGN = 50  # INCREASED from 5 to 50
SEQUENCE_LENGTH = 40  # frames per clip
LANDMARK_DIM = 258    # dimensions per frame

np.random.seed(42)

for sign_idx, sign_name in enumerate(SIGNS):
    # Create a unique pattern for each gesture (different hand positions)
    # Using sign_idx to create distinctive base patterns
    gesture_pattern = np.zeros(LANDMARK_DIM)

    # Hand landmarks are roughly at indices 33*4 to 33*4+42 (left+right hands)
    # Set different positions for each gesture
    hand_start = 132  # After pose landmarks
    gesture_pattern[hand_start:hand_start+21] = 0.3 + sign_idx * 0.1  # Left hand
    gesture_pattern[hand_start+21:hand_start+42] = 0.7 - sign_idx * 0.1  # Right hand

    for clip_num in range(1, NUM_CLIPS_PER_SIGN + 1):
        # Create sequence with meaningful variations
        sequence = []
        for frame_idx in range(SEQUENCE_LENGTH):
            # Start with gesture-specific pattern
            frame = gesture_pattern.copy()

            # Add realistic motion (hand moves over frames)
            motion = np.sin(np.pi * frame_idx / SEQUENCE_LENGTH) * 0.2
            frame[hand_start:hand_start+42] += motion

            # Add small noise for variation
            noise = np.random.randn(LANDMARK_DIM) * 0.03
            frame = frame + noise
            frame = np.clip(frame, 0, 1)
            sequence.append(frame)

        sequence_array = np.array(sequence, dtype=np.float32)

        # Save as .npy file
        output_path = DATA_DIR / f"{sign_name}_{clip_num:02d}.npy"
        np.save(output_path, sequence_array)
        if clip_num % 10 == 0:
            print(f"Generated {output_path}: shape {sequence_array.shape}")

print(f"\n✓ Generated {len(SIGNS) * NUM_CLIPS_PER_SIGN} synthetic landmark sequences")
print(f"  Signs: {SIGNS}")
print(f"  Samples per sign: {NUM_CLIPS_PER_SIGN}")
print(f"  Location: {DATA_DIR}")