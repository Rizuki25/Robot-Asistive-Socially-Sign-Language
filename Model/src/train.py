"""
=============================================================
Tahap 3: Training Model BiLSTM
=============================================================
Melatih model BiLSTM menggunakan PyTorch.
Fitur: early stopping, learning rate scheduler, logging.

Input:  dataset/processed/ (train/val .pt files)
Output: outputs/models/best_model.pth
        outputs/logs/training_log.csv
        outputs/figures/loss_curve.png, accuracy_curve.png
=============================================================
"""

import os
import json
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import BiLSTMModel
from dataset_loader import create_dataloaders
from utils import set_seed, get_device, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device
) -> tuple:
    """
    Training untuk satu epoch.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistik
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device
) -> tuple:
    """
    Validasi model pada validation set.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_training_curves(log_df: pd.DataFrame, output_dir: str) -> None:
    """Plot loss dan accuracy curves dan simpan sebagai gambar."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(log_df["epoch"], log_df["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(log_df["epoch"], log_df["val_acc"], label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Training curves → {output_dir}/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Training model BiLSTM")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path ke file konfigurasi (default: configs/config.yaml)"
    )
    args = parser.parse_args()

    # Load konfigurasi
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed & device
    seed = config["training"]["random_seed"]
    set_seed(seed)
    device = get_device()

    # Buat output directories
    models_dir = config["paths"]["models"]
    logs_dir = config["paths"]["logs"]
    figures_dir = config["paths"]["figures"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load label encoder untuk mendapatkan num_classes
    encoder_path = os.path.join(config["paths"]["processed"], "label_encoder.json")
    with open(encoder_path, "r") as f:
        label_encoder = json.load(f)
    num_classes = label_encoder["num_classes"]

    print(f"{'='*60}")
    print(f"  Sign Language Recognition - BiLSTM Training")
    print(f"{'='*60}")
    print(f"  Device       : {device}")
    print(f"  Num classes   : {num_classes}")
    print(f"  Hidden size   : {config['model']['hidden_size']}")
    print(f"  Num layers    : {config['model']['num_layers']}")
    print(f"  Learning rate : {config['training']['learning_rate']}")
    print(f"  Batch size    : {config['training']['batch_size']}")
    print(f"  Max epochs    : {config['training']['epochs']}")
    print(f"{'='*60}")

    # Buat DataLoaders
    print("\n[INFO] Loading data...")
    dataloaders = create_dataloaders(
        processed_dir=config["paths"]["processed"],
        batch_size=config["training"]["batch_size"]
    )

    # Inisialisasi model
    model = BiLSTMModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"]
    ).to(device)

    print(f"\n{model.get_model_summary()}")

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["patience"]
    training_log = []

    print("\n[INFO] Memulai training...\n")
    start_time = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, device
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        training_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr
        })

        print(
            f"  Epoch [{epoch:3d}/{config['training']['epochs']}] "
            f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.6f}"
        )

        # Early stopping & save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(models_dir, "best_model.pth")
            )
            print(f"    ✓ Model terbaik disimpan (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  [EARLY STOP] Tidak ada improvement selama {patience} epoch.")
                break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training selesai dalam {elapsed:.1f} detik ({elapsed/60:.1f} menit)")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    # Simpan training log
    log_df = pd.DataFrame(training_log)
    log_path = os.path.join(logs_dir, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\n  [SAVED] Training log → {log_path}")

    # Plot training curves
    plot_training_curves(log_df, figures_dir)


if __name__ == "__main__":
    main()
