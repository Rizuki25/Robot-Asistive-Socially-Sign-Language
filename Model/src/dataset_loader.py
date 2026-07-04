"""
=============================================================
PyTorch Dataset & DataLoader
=============================================================
Custom Dataset class untuk memuat data landmark yang telah
diproses (file .pt) ke dalam PyTorch DataLoader.
=============================================================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class SignLanguageDataset(Dataset):
    """
    Custom PyTorch Dataset untuk data bahasa isyarat.

    Args:
        data_path (str): Path ke file data .pt (e.g., train_data.pt)
        labels_path (str): Path ke file labels .pt (e.g., train_labels.pt)
    """

    def __init__(self, data_path: str, labels_path: str):
        self.data = torch.load(data_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)

        assert len(self.data) == len(self.labels), (
            f"Jumlah data ({len(self.data)}) dan labels ({len(self.labels)}) tidak sama!"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx], self.labels[idx]


def create_dataloaders(
    processed_dir: str,
    batch_size: int = 32,
    num_workers: int = 0
) -> dict:
    """
    Buat DataLoader untuk train, validation, dan test set.

    Args:
        processed_dir: Path ke folder dataset/processed/
        batch_size: Ukuran batch
        num_workers: Jumlah worker untuk data loading

    Returns:
        dict: Dictionary berisi DataLoader untuk train, val, dan test
    """
    dataloaders = {}

    for split in ["train", "val", "test"]:
        data_path = os.path.join(processed_dir, f"{split}_data.pt")
        labels_path = os.path.join(processed_dir, f"{split}_labels.pt")

        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            print(f"[WARN] File tidak ditemukan untuk split '{split}', skip.")
            continue

        dataset = SignLanguageDataset(data_path, labels_path)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # Shuffle hanya untuk training
            num_workers=num_workers,
            drop_last=False
        )

        print(f"  [{split.upper()}] {len(dataset)} samples, "
              f"{len(dataloaders[split])} batches (batch_size={batch_size})")

    return dataloaders
