# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
from data_loader import LandmarkDataset
from model import LSTMClassifier

# Configuration
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/landmarks"
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

# Load dataset
dataset = LandmarkDataset(data_dir=DATA_DIR, seq_len=40)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Class mapping: {dataset.label_to_idx}")

# Create model
model = LSTMClassifier(
    input_dim=258,
    hidden_dim=128,
    num_layers=2,
    num_classes=dataset.num_classes,
    dropout=0.3
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
print("\n" + "="*60)
print("Starting training...")
print("="*60)

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    if (epoch) % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_DIR / "model.pt")
        print(f"           ✓ Saved best model (loss: {best_loss:.4f})")

print("\n" + "="*60)
print("✓ Training complete!")
print("="*60)

# Save label mapping
labels_map = {str(k): v for k, v in dataset.idx_to_label.items()}
with open(MODEL_DIR / "labels_map.json", "w") as f:
    json.dump(labels_map, f, indent=2)

print(f"✓ Saved model to {MODEL_DIR / 'model.pt'}")
print(f"✓ Saved labels to {MODEL_DIR / 'labels_map.json'}")