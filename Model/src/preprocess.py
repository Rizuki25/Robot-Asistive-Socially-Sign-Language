"""
=============================================================
Tahap 2: Preprocessing Data
=============================================================
Mempersiapkan data landmark agar siap dilatih oleh model BiLSTM.
Meliputi normalisasi, padding/truncating sequence, label encoding,
dan pembagian data menjadi train/val/test set.

Input:  dataset/landmarks/{nama_kelas}/video.npy
Output: dataset/processed/
        ├── train_data.pt, train_labels.pt
        ├── val_data.pt, val_labels.pt
        ├── test_data.pt, test_labels.pt
        └── label_encoder.json
=============================================================
"""

import os
import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_landmarks(input_dir: str) -> tuple:
    """
    Load semua file landmark .npy dari folder dataset/landmarks/.

    Args:
        input_dir: Path ke folder landmarks

    Returns:
        tuple: (list_of_sequences, list_of_labels, label_to_idx_dict)
    """
    sequences = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"[INFO] Label mapping: {label_to_idx}")

    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        npy_files = sorted([
            f for f in os.listdir(class_dir)
            if f.endswith('.npy')
        ])

        for npy_file in tqdm(npy_files, desc=f"  Loading {class_name}"):
            filepath = os.path.join(class_dir, npy_file)
            landmark_data = np.load(filepath)  # Shape: (num_frames, 21, 3)

            if landmark_data.size == 0:
                continue

            sequences.append(landmark_data)
            labels.append(label_to_idx[class_name])

    return sequences, labels, label_to_idx


def normalize_data(sequences: list, method: str = "minmax") -> list:
    """
    Normalisasi data landmark.

    Args:
        sequences: List of numpy arrays
        method: "minmax" atau "zscore"

    Returns:
        list: Normalized sequences
    """
    normalized = []

    if method == "minmax":
        for seq in sequences:
            seq_min = seq.min()
            seq_max = seq.max()
            if seq_max - seq_min > 0:
                seq_norm = (seq - seq_min) / (seq_max - seq_min)
            else:
                seq_norm = seq
            normalized.append(seq_norm)

    elif method == "zscore":
        for seq in sequences:
            mean = seq.mean()
            std = seq.std()
            if std > 0:
                seq_norm = (seq - mean) / std
            else:
                seq_norm = seq
            normalized.append(seq_norm)

    else:
        raise ValueError(f"Metode normalisasi tidak dikenal: {method}")

    return normalized


def pad_or_truncate(sequences: list, max_seq_length: int) -> np.ndarray:
    """
    Padding atau truncating sequence agar panjang seragam.
    Juga flatten landmark dari (21, 3) menjadi (63,).

    Args:
        sequences: List of numpy arrays dengan shape (num_frames, 21, 3)
        max_seq_length: Panjang sequence target

    Returns:
        np.ndarray: Array dengan shape (num_samples, max_seq_length, 63)
    """
    processed = []

    for seq in sequences:
        # Flatten: (num_frames, 21, 3) → (num_frames, 63)
        num_frames = seq.shape[0]
        seq_flat = seq.reshape(num_frames, -1)  # (num_frames, 63)

        if num_frames >= max_seq_length:
            # Truncate: ambil max_seq_length frame pertama
            seq_padded = seq_flat[:max_seq_length]
        else:
            # Padding: tambahkan zeros di akhir
            padding = np.zeros((max_seq_length - num_frames, seq_flat.shape[1]))
            seq_padded = np.vstack([seq_flat, padding])

        processed.append(seq_padded)

    return np.array(processed, dtype=np.float32)


def split_data(
    data: np.ndarray,
    labels: list,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> dict:
    """
    Bagi data menjadi train, validation, dan test set.

    Args:
        data: Array data (num_samples, max_seq_length, 63)
        labels: List of integer labels
        test_size: Proporsi test set
        val_size: Proporsi validation set
        random_seed: Random seed

    Returns:
        dict: Dictionary berisi split data dan labels
    """
    labels = np.array(labels)

    # Split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data, labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )

    # Split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y_train_val
    )

    return {
        "train_data": X_train,
        "train_labels": y_train,
        "val_data": X_val,
        "val_labels": y_val,
        "test_data": X_test,
        "test_labels": y_test,
    }


def save_processed_data(split_data: dict, label_to_idx: dict, output_dir: str) -> None:
    """
    Simpan data yang sudah diproses sebagai file PyTorch .pt dan label encoder JSON.

    Args:
        split_data: Dictionary berisi split data
        label_to_idx: Dictionary mapping label name → index
        output_dir: Path output folder
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, value in split_data.items():
        tensor = torch.tensor(value, dtype=torch.float32 if "data" in key else torch.long)
        filepath = os.path.join(output_dir, f"{key}.pt")
        torch.save(tensor, filepath)
        print(f"  [SAVED] {filepath} — shape: {tensor.shape}")

    # Simpan label encoder
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    label_encoder = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "num_classes": len(label_to_idx)
    }
    encoder_path = os.path.join(output_dir, "label_encoder.json")
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump(label_encoder, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {encoder_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing data landmark untuk training BiLSTM"
    )
    parser.add_argument(
        "--input_dir", type=str, default="dataset/landmarks",
        help="Path ke folder landmark (default: dataset/landmarks)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/processed",
        help="Path ke folder output (default: dataset/processed)"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=50,
        help="Panjang sequence maksimum (default: 50)"
    )
    parser.add_argument(
        "--normalization", type=str, default="minmax",
        choices=["minmax", "zscore"],
        help="Metode normalisasi (default: minmax)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.15,
        help="Proporsi data test (default: 0.15)"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.15,
        help="Proporsi data validation (default: 0.15)"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    print("[INFO] Memulai preprocessing data...")
    print(f"  Input          : {args.input_dir}")
    print(f"  Output         : {args.output_dir}")
    print(f"  Max seq length : {args.max_seq_length}")
    print(f"  Normalisasi    : {args.normalization}")
    print(f"  Test size      : {args.test_size}")
    print(f"  Val size       : {args.val_size}")

    # 1. Load landmark data
    print("\n[STEP 1] Loading landmark data...")
    sequences, labels, label_to_idx = load_landmarks(args.input_dir)
    print(f"  Total samples: {len(sequences)}")

    if len(sequences) == 0:
        print("[ERROR] Tidak ada data landmark ditemukan!")
        return

    # 2. Normalisasi
    print(f"\n[STEP 2] Normalisasi ({args.normalization})...")
    sequences = normalize_data(sequences, method=args.normalization)

    # 3. Padding / Truncating
    print(f"\n[STEP 3] Padding/Truncating (max_seq_length={args.max_seq_length})...")
    data = pad_or_truncate(sequences, args.max_seq_length)
    print(f"  Data shape: {data.shape}")

    # 4. Split data
    print(f"\n[STEP 4] Splitting data (train/val/test)...")
    splits = split_data(
        data, labels,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.random_seed
    )
    print(f"  Train : {splits['train_data'].shape[0]} samples")
    print(f"  Val   : {splits['val_data'].shape[0]} samples")
    print(f"  Test  : {splits['test_data'].shape[0]} samples")

    # 5. Simpan
    print(f"\n[STEP 5] Menyimpan data processed...")
    save_processed_data(splits, label_to_idx, args.output_dir)

    print(f"\n{'='*50}")
    print("[DONE] Preprocessing selesai!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
