# quick_predict.py
import torch
import numpy as np
import json
from pathlib import Path
from model import LSTMClassifier
import glob

MODEL_DIR = Path("data/models")
model_path = MODEL_DIR / "model.pt"
labels_path = MODEL_DIR / "labels_map.json"

if not model_path.exists():
    print("❌ Model not found. Run train.py first.")
    exit(1)

# Load labels
with open(labels_path) as f:
    idx_to_label = json.load(f)

num_classes = len(idx_to_label)
model = LSTMClassifier(input_dim=258, hidden_dim=128, num_layers=2, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("✓ Model loaded")
print(f"Labels: {idx_to_label}\n")

# Test on a random sample
npy_files = glob.glob("data/landmarks/*.npy")
if npy_files:
    test_file = npy_files[0]
    print(f"Testing on: {test_file}")

    arr = np.load(test_file)
    seq_len = 40
    if arr.shape[0] >= seq_len:
        seq = arr[:seq_len]
    else:
        pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
        seq = np.vstack([arr, pad])

    x = torch.tensor(seq[None, ...], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        pred_idx = int(probs.argmax())
        pred_label = idx_to_label[str(pred_idx)]
        confidence = float(probs.max())

    print(f"Predicted: {pred_label} ({confidence:.2%})")
    print(f"\nAll probabilities:")
    for idx, label in idx_to_label.items():
        print(f"  {label}: {probs[int(idx)]:.3f}")