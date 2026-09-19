# export_onnx.py
import torch
import json
from pathlib import Path
from model import LSTMClassifier

MODEL_DIR = Path("data/models")
model_path = MODEL_DIR / "model.pt"
labels_path = MODEL_DIR / "labels_map.json"
onnx_path = MODEL_DIR / "sign_language_model.onnx"

if not model_path.exists():
    print("❌ Model not found. Run train.py first.")
    exit(1)

# Load labels to get num_classes
with open(labels_path) as f:
    idx_to_label = json.load(f)

num_classes = len(idx_to_label)

# Create model and load weights
model = LSTMClassifier(input_dim=258, hidden_dim=128, num_layers=2, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Create dummy input (batch_size=1, seq_len=40, input_dim=258)
dummy_input = torch.randn(1, 40, 258)

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=14,
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size"}
    }
)

print(f"✓ Exported to {onnx_path}")
print(f"  Input shape: (batch_size, 40, 258)")
print(f"  Output shape: (batch_size, {num_classes})")
print(f"  Classes: {list(idx_to_label.values())}")