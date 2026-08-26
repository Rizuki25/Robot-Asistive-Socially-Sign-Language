"""
Evaluasi model gabungan 36 kelas (Huruf & Kata) pada test set.

Jalankan dari folder Model/:
    python -m src.combined.evaluate --config configs/combined_90.yaml
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import yaml

from src.common.utils import get_device
from src.combined.dataset_loader import get_combined_data_loaders
from src.combined.model import CombinedBiLSTM


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str) -> None:
    """Plot confusion matrix 36 kelas beresolusi tinggi."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix - Combined Model (36 Classes)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix tersimpan: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi Combined 36-Class Model")
    parser.add_argument("--config", default="configs/combined_90.yaml", help="Path config gabungan")
    parser.add_argument("--model_path", default=None, help="Path model checkpoint (default: best_model.pth)")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Config tidak ditemukan: {args.config}")
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()
    paths = config["paths"]

    enc_path = os.path.join(paths["processed"], "label_encoder.json")
    with open(enc_path, "r", encoding="utf-8") as f:
        encoder = json.load(f)

    num_classes = encoder["num_classes"]
    class_names = encoder["classes"]
    letters_classes = set(encoder.get("letters_classes", []))
    words_classes = set(encoder.get("words_classes", []))

    model_path = args.model_path or os.path.join(paths["models"], "best_model.pth")
    if not os.path.isfile(model_path):
        print(f"[ERROR] Checkpoint model tidak ditemukan: {model_path}")
        return 1

    _, _, test_loader = get_combined_data_loaders(config)

    model = CombinedBiLSTM(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, lengths, labels in test_loader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            outputs = model(inputs, lengths)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrik Total
    acc = float(np.mean(all_preds == all_targets))
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    macro_f1 = float(precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)[2])

    # Analisis Sub-Kategori (Huruf vs Kata)
    letter_indices = [i for i, name in enumerate(class_names) if name in letters_classes]
    word_indices = [i for i, name in enumerate(class_names) if name in words_classes]

    is_true_letter = np.isin(all_targets, letter_indices)
    is_true_word = np.isin(all_targets, word_indices)

    is_pred_letter = np.isin(all_preds, letter_indices)
    is_pred_word = np.isin(all_preds, word_indices)

    letter_acc = float(np.mean(all_preds[is_true_letter] == all_targets[is_true_letter])) if np.sum(is_true_letter) > 0 else 0.0
    word_acc = float(np.mean(all_preds[is_true_word] == all_targets[is_true_word])) if np.sum(is_true_word) > 0 else 0.0

    # Cross-Confusion (Huruf tertukar jadi Kata, atau Kata tertukar jadi Huruf)
    letters_confused_as_words = int(np.sum(is_true_letter & is_pred_word))
    words_confused_as_letters = int(np.sum(is_true_word & is_pred_letter))

    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\n=======================================================")
    print("HASIL EVALUASI COMBINED MODEL (36 KELAS)")
    print("=======================================================")
    print(f"Total Test Samples     : {len(all_targets)}")
    print(f"Overall Accuracy       : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Weighted Precision     : {precision:.4f}")
    print(f"Weighted Recall        : {recall:.4f}")
    print(f"Weighted F1-Score      : {f1:.4f}")
    print(f"Macro F1-Score         : {macro_f1:.4f}")
    print("-------------------------------------------------------")
    print("SUB-CATEGORY PERFORMANCE:")
    print(f"  Akurasi Huruf (26 kelas) : {letter_acc * 100:.2f}% ({np.sum(is_true_letter)} sampel)")
    print(f"  Akurasi Kata  (10 kelas) : {word_acc * 100:.2f}% ({np.sum(is_true_word)} sampel)")
    print("CROSS-CONFUSION:")
    print(f"  Huruf tertukar jadi Kata : {letters_confused_as_words} sampel ({letters_confused_as_words / max(1, np.sum(is_true_letter)) * 100:.2f}%)")
    print(f"  Kata tertukar jadi Huruf : {words_confused_as_letters} sampel ({words_confused_as_letters / max(1, np.sum(is_true_word)) * 100:.2f}%)")
    print("-------------------------------------------------------")
    print("\nClassification Report:\n")
    print(report)

    # Simpan hasil
    os.makedirs(paths["results"], exist_ok=True)
    os.makedirs(paths["figures"], exist_ok=True)

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    plot_confusion_matrix(cm, class_names, os.path.join(paths["figures"], "confusion_matrix.png"))

    with open(os.path.join(paths["results"], "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("HASIL EVALUASI COMBINED 36-CLASS MODEL\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Letters Sub-Accuracy: {letter_acc * 100:.2f}%\n")
        f.write(f"Words Sub-Accuracy: {word_acc * 100:.2f}%\n")
        f.write(f"Cross-Confusion Letters->Words: {letters_confused_as_words}\n")
        f.write(f"Cross-Confusion Words->Letters: {words_confused_as_letters}\n\n")
        f.write(report)

    metrics_dict = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "letter_accuracy": letter_acc,
        "word_accuracy": word_acc,
        "letters_confused_as_words": letters_confused_as_words,
        "words_confused_as_letters": words_confused_as_letters,
        "total_test_samples": len(all_targets),
        "model_path": model_path,
    }
    with open(os.path.join(paths["results"], "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

