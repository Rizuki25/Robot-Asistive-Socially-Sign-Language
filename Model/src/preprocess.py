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


LANDMARKS_PER_HAND = 21


def normalize_data(sequences: list, method: str = "wrist_relative") -> list:
    """
    Normalisasi data landmark.

    Args:
        sequences: List of numpy arrays, tiap elemen shape (num_frames, 21*num_hands, 3)
                   (num_hands=1 untuk huruf, 2 untuk kata)
        method: "wrist_relative", "minmax", atau "zscore"

    Returns:
        list: Normalized sequences (shape sama seperti input)
    """
    normalized = []

    if method == "wrist_relative":
        # Translation-invariant: center tiap tangan ke wrist-nya sendiri (landmark 0 dalam blok)
        # Scale-invariant: skala dengan jarak wrist -> middle_finger_mcp (landmark 9 dalam blok)
        # Diterapkan per-blok 21 landmark supaya mendukung 1 tangan (huruf) maupun 2 tangan (kata).
        for seq in sequences:
            num_landmarks_total = seq.shape[1]
            assert num_landmarks_total % LANDMARKS_PER_HAND == 0, (
                f"Jumlah landmark ({num_landmarks_total}) harus kelipatan {LANDMARKS_PER_HAND}"
            )
            num_hands = num_landmarks_total // LANDMARKS_PER_HAND

            seq_norm = np.zeros_like(seq)
            for h in range(num_hands):
                start = h * LANDMARKS_PER_HAND
                end = start + LANDMARKS_PER_HAND
                hand_block = seq[:, start:end, :]  # (num_frames, 21, 3)

                wrist = hand_block[:, 0:1, :]  # (num_frames, 1, 3)
                centered = hand_block - wrist

                ref_point = centered[:, 9, :]  # (num_frames, 3)
                ref_dist = np.linalg.norm(ref_point, axis=-1)  # (num_frames,)
                ref_dist = np.where(ref_dist > 1e-6, ref_dist, 1.0)

                seq_norm[:, start:end, :] = centered / ref_dist[:, None, None]

            normalized.append(seq_norm)

    elif method == "minmax":
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


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Hasilkan satu varian teraugmentasi dari sequence landmark (sudah dinormalisasi
    wrist-relative). Semua transformasi menjaga identitas kelas (bukan mirror-flip):
    rotasi kecil, scale jitter, gaussian noise, dan time-warp (variasi kecepatan).

    Args:
        seq: Array (num_frames, 21, 3), hasil normalize_data(method="wrist_relative")
        rng: np.random.Generator untuk random reproducible

    Returns:
        np.ndarray: Sequence teraugmentasi, shape (num_frames_baru, 21, 3)
    """
    num_frames = seq.shape[0]

    # 1. Time-warp: resample sumbu waktu dengan faktor kecepatan acak
    speed_factor = rng.uniform(0.85, 1.15)
    new_len = max(2, int(round(num_frames * speed_factor)))
    orig_idx = np.linspace(0, num_frames - 1, num=num_frames)
    new_idx = np.linspace(0, num_frames - 1, num=new_len)
    warped = np.empty((new_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
    for lm in range(seq.shape[1]):
        for coord in range(seq.shape[2]):
            warped[:, lm, coord] = np.interp(new_idx, orig_idx, seq[:, lm, coord])

    # 2. Rotasi kecil di bidang x,y (sekitar origin/wrist)
    angle = np.deg2rad(rng.uniform(-15, 15))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    xy = warped[:, :, :2]
    warped[:, :, :2] = xy @ rot_matrix.T

    # 3. Scale jitter
    scale = rng.uniform(0.85, 1.15)
    warped = warped * scale

    # 4. Gaussian noise kecil
    noise = rng.normal(loc=0.0, scale=0.01, size=warped.shape)
    warped = warped + noise

    return warped.astype(np.float32)


def augment_train_set(sequences: list, labels: list, factor: int = 3, random_seed: int = 42) -> tuple:
    """
    Perbanyak training set dengan menambahkan `factor` salinan teraugmentasi
    per sample asli. Data asli tetap disertakan (tidak diganti).

    Args:
        sequences: List of arrays (num_frames, 21, 3), sudah dinormalisasi
        labels: List of int label, sejajar dengan sequences
        factor: Jumlah salinan augmentasi per sample asli
        random_seed: Seed untuk reproducibility

    Returns:
        tuple: (augmented_sequences, augmented_labels)
    """
    rng = np.random.default_rng(random_seed)

    all_sequences = list(sequences)
    all_labels = list(labels)

    for seq, label in zip(sequences, labels):
        for _ in range(factor):
            all_sequences.append(augment_sequence(seq, rng))
            all_labels.append(label)

    return all_sequences, all_labels


def split_indices(
    labels: list,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> dict:
    """
    Bagi index sample menjadi train, validation, dan test set (stratified).

    Args:
        labels: List of integer labels
        test_size: Proporsi test set
        val_size: Proporsi validation set
        random_seed: Random seed

    Returns:
        dict: {"train": idx_array, "val": idx_array, "test": idx_array}
    """
    labels = np.array(labels)
    all_idx = np.arange(len(labels))

    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )

    val_ratio = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=labels[train_val_idx]
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


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
        "--input_dir", type=str, default="dataset/letters/landmarks",
        help="Path ke folder landmark (default: dataset/letters/landmarks)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/letters/processed",
        help="Path ke folder output (default: dataset/letters/processed)"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=90,
        help="Panjang sequence maksimum (default: 90)"
    )
    parser.add_argument(
        "--normalization", type=str, default="wrist_relative",
        choices=["wrist_relative", "minmax", "zscore"],
        help="Metode normalisasi (default: wrist_relative)"
    )
    parser.add_argument(
        "--augment_factor", type=int, default=3,
        help="Jumlah salinan augmentasi per sample training (default: 3, 0 = nonaktif)"
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
    print(f"  Augment factor : {args.augment_factor}")
    print(f"  Test size      : {args.test_size}")
    print(f"  Val size       : {args.val_size}")

    # 1. Load landmark data
    print("\n[STEP 1] Loading landmark data...")
    sequences, labels, label_to_idx = load_landmarks(args.input_dir)
    print(f"  Total samples: {len(sequences)}")

    if len(sequences) == 0:
        print("[ERROR] Tidak ada data landmark ditemukan!")
        return

    # 2. Split index (sebelum normalisasi/augmentasi, agar train/val/test independen)
    print(f"\n[STEP 2] Splitting index (train/val/test)...")
    idx_splits = split_indices(
        labels,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.random_seed
    )

    labels_arr = np.array(labels)
    split_sequences = {
        split: [sequences[i] for i in idx]
        for split, idx in idx_splits.items()
    }
    split_labels = {
        split: labels_arr[idx].tolist()
        for split, idx in idx_splits.items()
    }
    print(f"  Train : {len(split_sequences['train'])} samples")
    print(f"  Val   : {len(split_sequences['val'])} samples")
    print(f"  Test  : {len(split_sequences['test'])} samples")

    # 3. Normalisasi per split
    print(f"\n[STEP 3] Normalisasi ({args.normalization})...")
    for split in ["train", "val", "test"]:
        split_sequences[split] = normalize_data(split_sequences[split], method=args.normalization)

    # 4. Augmentasi (hanya pada train split)
    if args.augment_factor > 0:
        print(f"\n[STEP 4] Augmentasi train set (factor={args.augment_factor})...")
        split_sequences["train"], split_labels["train"] = augment_train_set(
            split_sequences["train"], split_labels["train"],
            factor=args.augment_factor, random_seed=args.random_seed
        )
        print(f"  Train (setelah augmentasi): {len(split_sequences['train'])} samples")

    # 5. Padding / Truncating per split
    print(f"\n[STEP 5] Padding/Truncating (max_seq_length={args.max_seq_length})...")
    splits = {}
    for split in ["train", "val", "test"]:
        splits[f"{split}_data"] = pad_or_truncate(split_sequences[split], args.max_seq_length)
        splits[f"{split}_labels"] = np.array(split_labels[split])
        print(f"  {split.capitalize():5s}: {splits[f'{split}_data'].shape}")

    # 6. Simpan
    print(f"\n[STEP 6] Menyimpan data processed...")
    save_processed_data(splits, label_to_idx, args.output_dir)

    print(f"\n{'='*50}")
    print("[DONE] Preprocessing selesai!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
