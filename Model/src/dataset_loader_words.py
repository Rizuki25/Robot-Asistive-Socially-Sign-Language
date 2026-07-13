"""Dataset loader khusus pipeline kata motion-aware."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class WordMotionDataset(Dataset):
    def __init__(self, processed_dir: str, split: str):
        root = Path(processed_dir)
        self.data = torch.load(root / f"{split}_data.pt", weights_only=True)
        self.labels = torch.load(root / f"{split}_labels.pt", weights_only=True)
        self.lengths = torch.load(root / f"{split}_lengths.pt", weights_only=True)

        if not (len(self.data) == len(self.labels) == len(self.lengths)):
            raise ValueError(f"Jumlah data/label/length split {split} tidak sama")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.lengths[index]


def create_word_dataloaders(processed_dir: str, batch_size: int, num_workers: int = 0):
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = WordMotionDataset(processed_dir, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            drop_last=False,
        )
        print(f"  [{split.upper()}] {len(dataset)} samples, {len(loaders[split])} batches")
    return loaders
