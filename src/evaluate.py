"""
=============================================================
Tahap 4: Evaluasi Model
=============================================================
Mengevaluasi performa model BiLSTM pada test set.
Metrik: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

Input:  outputs/models/best_model.pth + dataset/processed/test_*.pt
Output: outputs/results/classification_report.txt
        outputs/results/evaluation_metrics.json
        outputs/figures/confusion_matrix.png
=============================================================
"""

import os
import json
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import BiLSTMModel
from dataset_loader import create_dataloaders
from utils import get_device


def evaluate_model(model, dataloader, device) -> tuple:
    """
    Evaluasi model pada dataloader.

    Returns:
        tuple: (all_predictions, all_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str
) -> None:
    """Plot dan simpan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Confusion matrix → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model BiLSTM")
    parser.add_argument(
        "--model_path", type=str, default="outputs/models/best_model.pth",
        help="Path ke model terbaik (default: outputs/models/best_model.pth)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path ke file konfigurasi (default: configs/config.yaml)"
    )
    args = parser.parse_args()

    # Load konfigurasi
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device()

    # Load label encoder
    encoder_path = os.path.join(config["paths"]["processed"], "label_encoder.json")
    with open(encoder_path, "r") as f:
        label_encoder = json.load(f)

    num_classes = label_encoder["num_classes"]
    idx_to_label = label_encoder["idx_to_label"]
    # JSON keys are strings, convert to int
    class_names = [idx_to_label[str(i)] for i in range(num_classes)]

    print(f"{'='*60}")
    print(f"  Sign Language Recognition - Model Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model_path}")
    print(f"  Classes    : {class_names}")
    print(f"{'='*60}")

    # Inisialisasi model
    model = BiLSTMModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"]
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n  Model loaded dari epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['val_loss']:.4f}, val_acc: {checkpoint['val_acc']:.4f})")

    # Load test data
    print("\n[INFO] Loading test data...")
    dataloaders = create_dataloaders(
        processed_dir=config["paths"]["processed"],
        batch_size=config["training"]["batch_size"]
    )

    if "test" not in dataloaders:
        print("[ERROR] Test dataloader tidak ditemukan!")
        return

    # Evaluasi
    print("\n[INFO] Evaluating on test set...")
    y_pred, y_true = evaluate_model(model, dataloaders["test"], device)

    # Hitung metrik
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*40}")
    print(f"  HASIL EVALUASI")
    print(f"{'='*40}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"{'='*40}")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(f"\n  Classification Report:\n{report}")

    # Buat output directories
    results_dir = config["paths"]["results"]
    figures_dir = config["paths"]["figures"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Simpan classification report
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Sign Language Recognition - BiLSTM Evaluation Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  Precision : {precision:.4f}\n")
        f.write(f"  Recall    : {recall:.4f}\n")
        f.write(f"  F1-Score  : {f1:.4f}\n\n")
        f.write(f"Classification Report:\n{report}\n")
    print(f"  [SAVED] Classification report → {report_path}")

    # Simpan metrik sebagai JSON
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "model_path": args.model_path,
        "epoch": checkpoint["epoch"],
        "num_test_samples": len(y_true),
        "num_classes": num_classes,
        "class_names": class_names
    }
    metrics_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] Evaluation metrics → {metrics_path}")

    # Plot confusion matrix
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    print(f"\n{'='*60}")
    print(f"  Evaluasi selesai!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
