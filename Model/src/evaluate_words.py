"""Evaluasi khusus model kata motion-aware."""

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset_loader_words import create_word_dataloaders
from model_words import WordMotionBiLSTM
from utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model kata motion-aware")
    parser.add_argument("--config", default="configs/words_motion.yaml")
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    paths = config["paths"]
    model_path = args.model_path or os.path.join(paths["models"], "best_model.pth")
    device = get_device()

    with open(os.path.join(paths["processed"], "label_encoder.json"), encoding="utf-8") as file:
        encoder = json.load(file)
    class_names = [encoder["idx_to_label"][str(idx)] for idx in range(encoder["num_classes"])]

    model = WordMotionBiLSTM(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=encoder["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if checkpoint.get("pipeline") != "words_motion_v1":
        raise ValueError("Checkpoint bukan berasal dari pipeline words_motion_v1")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = create_word_dataloaders(
        paths["processed"],
        config["training"]["batch_size"],
    )["test"]
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            outputs = model(inputs.to(device), lengths.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    labels_range = list(range(encoder["num_classes"]))
    accuracy = accuracy_score(targets, predictions)
    weighted_precision = precision_score(
        targets, predictions, labels=labels_range, average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        targets, predictions, labels=labels_range, average="weighted", zero_division=0
    )
    weighted_f1 = f1_score(
        targets, predictions, labels=labels_range, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        targets, predictions, labels=labels_range, average="macro", zero_division=0
    )
    report = classification_report(
        targets,
        predictions,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
    )

    print("\n" + "=" * 48)
    print("HASIL EVALUASI WORDS MOTION")
    print("=" * 48)
    print(f"Accuracy           : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted precision : {weighted_precision:.4f}")
    print(f"Weighted recall    : {weighted_recall:.4f}")
    print(f"Weighted F1        : {weighted_f1:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}\n")
    print(report)

    os.makedirs(paths["results"], exist_ok=True)
    os.makedirs(paths["figures"], exist_ok=True)
    with open(os.path.join(paths["results"], "classification_report.txt"), "w", encoding="utf-8") as file:
        file.write(report)

    prediction_counts = Counter(predictions)
    metrics = {
        "pipeline": "words_motion_v1",
        "model_path": model_path,
        "epoch": checkpoint["epoch"],
        "num_test_samples": len(targets),
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "prediction_distribution": {
            class_names[idx]: prediction_counts.get(idx, 0) for idx in labels_range
        },
    }
    with open(os.path.join(paths["results"], "evaluation_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    matrix = confusion_matrix(targets, predictions, labels=labels_range)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Words Motion")
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths["figures"], "confusion_matrix.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[DONE] Hasil tersimpan di {paths['results']}")


if __name__ == "__main__":
    main()
