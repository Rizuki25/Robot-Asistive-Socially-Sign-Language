"""
Training pipeline untuk model gabungan 36 kelas (Huruf & Kata).

Jalankan dari folder Model/:
    python -m src.combined.train --config configs/combined_90.yaml
"""

import argparse
import csv
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import yaml

from src.common.utils import count_parameters, get_device
from src.combined.dataset_loader import get_combined_data_loaders
from src.combined.model import CombinedBiLSTM


def plot_training_curves(log_csv: str, output_path: str) -> None:
    """Plot loss dan akurasi per epoch."""
    epochs, train_losses, val_losses = [], [], []
    train_accs, val_accs = [], []

    with open(log_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_accs.append(float(row["train_acc"]))
            val_accs.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="darkorange", linewidth=2)
    ax1.set_title("Combined Model Loss Curves (36 Classes)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, [acc * 100 for acc in train_accs], label="Train Acc", color="royalblue", linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in val_accs], label="Val Acc", color="darkorange", linewidth=2)
    ax2.set_title("Combined Model Accuracy Curves (36 Classes)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Grafik training tersimpan: {output_path}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, lengths, labels in loader:
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, lengths, labels in loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Combined 36-Class Model")
    parser.add_argument("--config", default="configs/combined_90.yaml", help="Path config gabungan")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Config tidak ditemukan: {args.config}")
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()
    train_cfg = config["training"]
    paths = config["paths"]

    enc_path = os.path.join(paths["processed"], "label_encoder.json")
    if not os.path.isfile(enc_path):
        print(f"[ERROR] Label encoder tidak ditemukan di {enc_path}. Jalankan preprocess terlebih dahulu!")
        return 1

    with open(enc_path, "r", encoding="utf-8") as f:
        encoder = json.load(f)
    num_classes = encoder["num_classes"]

    train_loader, val_loader, _ = get_combined_data_loaders(config)

    model = CombinedBiLSTM(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
    ).to(device)

    param_info = count_parameters(model)
    print(f"[INFO] Arsitektur: {config['model']['name']} ({num_classes} kelas)")
    print(f"[INFO] Parameter  : {param_info['trainable']:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = StepLR(
        optimizer,
        step_size=train_cfg.get("scheduler_step", 15),
        gamma=train_cfg.get("scheduler_gamma", 0.5),
    )

    epochs = train_cfg.get("epochs", 60)
    patience = train_cfg.get("patience", 15)

    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)
    os.makedirs(paths["figures"], exist_ok=True)

    best_model_path = os.path.join(paths["models"], "best_model.pth")
    last_model_path = os.path.join(paths["models"], "last_model.pth")
    log_csv = os.path.join(paths["logs"], "training_log.csv")

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "elapsed_s"])

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n=======================================================")
    print(f"MULAI TRAINING COMBINED 36-CLASS ({epochs} Epochs)")
    print("=======================================================")

    total_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}", f"{elapsed:.1f}"])

        is_best = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            is_best = True
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "pipeline": "combined_v1",
                "num_classes": num_classes,
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1

        marker = " [*BEST]" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train: Loss={train_loss:.4f}, Acc={train_acc*100:5.1f}% | "
              f"Val: Loss={val_loss:.4f}, Acc={val_acc*100:5.1f}% | LR={current_lr:.5f} ({elapsed:4.1f}s){marker}")

        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping pada epoch {epoch} (tidak ada peningkatan selama {patience} epoch)")
            break

    # Simpan model terakhir
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "pipeline": "combined_v1",
        "num_classes": num_classes,
    }, last_model_path)

    total_time = time.time() - total_start
    print("\n=======================================================")
    print("TRAINING GABUNGAN SELESAI!")
    print(f"  Total Waktu : {total_time/60:.2f} menit")
    print(f"  Best Val Acc: {best_val_acc*100:.2f}% (Loss: {best_val_loss:.4f})")
    print(f"  Best Model  : {best_model_path}")
    print("=======================================================\n")

    plot_training_curves(log_csv, os.path.join(paths["figures"], "training_curves.png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

