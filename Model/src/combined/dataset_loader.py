"""
PyTorch Dataset dan DataLoader untuk model gabungan 36 kelas.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CombinedDataset(Dataset):
    """Dataset gabungan yang mengembalikan (data, length, label)."""

    def __init__(self, npz_path: str):
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"File dataset tidak ditemukan: {npz_path}")
        data_dict = np.load(npz_path)
        self.data = torch.from_numpy(data_dict["data"]).float()
        self.lengths = torch.from_numpy(data_dict["lengths"]).long()
        self.labels = torch.from_numpy(data_dict["labels"]).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx], self.lengths[idx], self.labels[idx]


def get_combined_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Buat DataLoader untuk train, val, dan test set.
    """
    proc_dir = config["paths"]["processed"]
    batch_size = config["training"].get("batch_size", 32)
    num_workers = config["training"].get("num_workers", 0)

    train_set = CombinedDataset(os.path.join(proc_dir, "train.npz"))
    val_set = CombinedDataset(os.path.join(proc_dir, "val.npz"))
    test_set = CombinedDataset(os.path.join(proc_dir, "test.npz"))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader

