# data_loader.py
import numpy as np
import glob
from pathlib import Path
from torch.utils.data import Dataset
import torch

class LandmarkDataset(Dataset):
    def __init__(self, data_dir="data/landmarks", seq_len=40):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.files = list(self.data_dir.glob("*.npy"))

        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")

        # Create label mapping from filename
        self.label_to_idx = {}
        self.idx_to_label = {}
        for file_path in self.files:
            label_name = file_path.stem.rsplit('_', 1)[0]  # e.g., "hello" from "hello_01"
            if label_name not in self.label_to_idx:
                idx = len(self.label_to_idx)
                self.label_to_idx[label_name] = idx
                self.idx_to_label[idx] = label_name

        print(f"Labels found: {self.label_to_idx}")
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        arr = np.load(file_path)  # shape (T, D) where T=40, D=258

        # Pad or truncate to seq_len
        if arr.shape[0] >= self.seq_len:
            seq = arr[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            seq = np.vstack([arr, pad])

        label_name = file_path.stem.rsplit('_', 1)[0]
        label_idx = self.label_to_idx[label_name]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)