"""Training khusus model kata motion-aware."""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.common.utils import get_device, set_seed
from src.words.dataset_loader import create_word_dataloaders
from src.words.model import WordMotionBiLSTM


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, labels, lengths in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_training_curves(log_frame: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(log_frame["epoch"], log_frame["train_loss"], label="Train")
    axes[0].plot(log_frame["epoch"], log_frame["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(log_frame["epoch"], log_frame["train_acc"], label="Train")
    axes[1].plot(log_frame["epoch"], log_frame["val_acc"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "training_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Training packed BiLSTM khusus kata")
    parser.add_argument("--config", default="configs/words_motion.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    set_seed(config["training"]["random_seed"])
    device = get_device()
    paths = config["paths"]
    for key in ("models", "logs", "figures"):
        os.makedirs(paths[key], exist_ok=True)

    with open(os.path.join(paths["processed"], "label_encoder.json"), encoding="utf-8") as file:
        encoder = json.load(file)
    with open(os.path.join(paths["processed"], "feature_metadata.json"), encoding="utf-8") as file:
        feature_metadata = json.load(file)

    input_size = config["model"]["input_size"]
    if input_size != feature_metadata["input_size"]:
        raise ValueError(
            f"Config input_size={input_size}, tetapi processed data memiliki "
            f"input_size={feature_metadata['input_size']}"
        )

    loaders = create_word_dataloaders(
        paths["processed"],
        config["training"]["batch_size"],
    )
    model = WordMotionBiLSTM(
        input_size=input_size,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=encoder["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)
    print(f"\n{model.summary()}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    log_rows = []
    start_time = time.time()
    model_path = os.path.join(paths["models"], "best_model.pth")

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=config["training"].get("gradient_clip"),
        )
        val_loss, val_acc = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
        )
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": learning_rate,
            }
        )
        print(
            f"Epoch [{epoch:3d}/{config['training']['epochs']}] "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f} | lr={learning_rate:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "pipeline": "words_motion_v1",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "input_size": input_size,
                    "num_classes": encoder["num_classes"],
                },
                model_path,
            )
            print(f"  [SAVED] Model terbaik: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                print("[EARLY STOP] Validation loss tidak membaik.")
                break

    log_frame = pd.DataFrame(log_rows)
    log_frame.to_csv(os.path.join(paths["logs"], "training_log.csv"), index=False)
    save_training_curves(log_frame, paths["figures"])
    elapsed = time.time() - start_time
    print(f"\n[DONE] Training selesai dalam {elapsed / 60:.1f} menit")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
