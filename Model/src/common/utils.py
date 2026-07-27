"""
=============================================================
Utility Functions
=============================================================
Fungsi-fungsi utilitas umum yang digunakan di seluruh project.
=============================================================
"""

import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seed untuk reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Deteksi dan return device terbaik yang tersedia (CUDA / CPU).

    Returns:
        torch.device: Device yang akan digunakan
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    filepath: str
) -> None:
    """
    Simpan model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Epoch saat ini
        val_loss: Validation loss
        val_acc: Validation accuracy
        filepath: Path untuk menyimpan checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model, optimizer=None, device=None):
    """
    Load model checkpoint.

    Args:
        filepath: Path ke checkpoint file
        model: PyTorch model (arsitektur harus sama)
        optimizer: Optional optimizer untuk melanjutkan training
        device: Device target

    Returns:
        dict: Checkpoint info (epoch, val_loss, val_acc)
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    info = {
        "epoch": checkpoint.get("epoch", 0),
        "val_loss": checkpoint.get("val_loss", 0),
        "val_acc": checkpoint.get("val_acc", 0),
    }

    print(f"[INFO] Checkpoint loaded dari epoch {info['epoch']} "
          f"(val_loss: {info['val_loss']:.4f}, val_acc: {info['val_acc']:.4f})")

    return info


def count_parameters(model) -> dict:
    """
    Hitung jumlah parameter model.

    Args:
        model: PyTorch model

    Returns:
        dict: Total dan trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }
