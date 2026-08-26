"""
Preprocessing pipeline untuk menggabungkan dataset Huruf (26 kelas) dan Kata (10 kata)
menjadi satu dataset terpadu 36 kelas dengan 140 fitur motion-aware.

Jalankan dari folder Model/:
    python -m src.combined.preprocess --config configs/combined_90.yaml
"""

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import yaml

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES = 3
COMBINED_INPUT_SIZE = 140


def hand_presence(sequence: np.ndarray) -> np.ndarray:
    """Deteksi keberadaan tangan untuk setiap frame dan setiap slot tangan."""
    presence = []
    for hand_idx in range(NUM_HANDS):
        start = hand_idx * LANDMARKS_PER_HAND
        block = sequence[:, start : start + LANDMARKS_PER_HAND, :]
        active = np.any(np.abs(block) > 1e-8, axis=(1, 2))
        presence.append(active)
    return np.column_stack(presence)


def extract_combined_features(sequence: np.ndarray) -> np.ndarray:
    """
    Ekstraksi 140 fitur motion-aware per frame:
    - 63 bentuk relatif pergelangan tangan (wrist-relative local shape)
    - 3 koordinat global pergelangan tangan (global wrist)
    - 3 perpindahan pergelangan tangan terhadap frame awal (wrist displacement)
    - 1 mask keberadaan tangan (presence mask)
    Untuk Hand 1 (70 fitur) + Hand 2 (70 fitur) = 140 fitur.
    """
    presence = hand_presence(sequence)
    hand_features = []

    for hand_idx in range(NUM_HANDS):
        start = hand_idx * LANDMARKS_PER_HAND
        block = sequence[:, start : start + LANDMARKS_PER_HAND, :]
        valid = presence[:, hand_idx]

        wrist = block[:, 0, :]
        centered = block - wrist[:, None, :]
        palm_scale = np.linalg.norm(centered[:, 9, :], axis=1)
        safe_scale = np.where((palm_scale > 1e-6) & valid, palm_scale, 1.0)
        local_shape = centered / safe_scale[:, None, None]

        local_shape[~valid] = 0.0
        global_wrist = wrist.copy()
        global_wrist[~valid] = 0.0

        displacement = np.zeros_like(global_wrist)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            reference = global_wrist[valid_indices[0]].copy()
            displacement[valid] = global_wrist[valid] - reference

        features = np.concatenate(
            [
                local_shape.reshape(sequence.shape[0], -1),
                global_wrist,
                displacement,
                valid.astype(np.float32)[:, None],
            ],
            axis=1,
        )
        hand_features.append(features)

    result = np.concatenate(hand_features, axis=1).astype(np.float32)
    if result.shape[1] != COMBINED_INPUT_SIZE:
        raise RuntimeError(f"Ukuran fitur {result.shape[1]} != {COMBINED_INPUT_SIZE}")
    return result


def pad_features(features_list: List[np.ndarray], max_seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad array fitur ke max_seq_length dan kembalikan panjang valid."""
    padded = np.zeros((len(features_list), max_seq_length, COMBINED_INPUT_SIZE), dtype=np.float32)
    lengths = np.zeros((len(features_list),), dtype=np.int64)

    for idx, feat in enumerate(features_list):
        seq_len = min(len(feat), max_seq_length)
        padded[idx, :seq_len, :] = feat[:seq_len]
        lengths[idx] = seq_len

    return padded, lengths


def load_raw_samples(
    letters_dir: str,
    words_dir: str,
) -> Tuple[List[np.ndarray], List[str], List[str], List[str]]:
    """Muat landmark npy dari direktori huruf dan kata."""
    sequences = []
    labels = []
    letters_classes = []
    words_classes = []

    # 1. Muat Huruf
    if os.path.isdir(letters_dir):
        letter_folders = sorted([
            d for d in os.listdir(letters_dir)
            if os.path.isdir(os.path.join(letters_dir, d))
        ])
        for label in letter_folders:
            folder_path = os.path.join(letters_dir, label)
            files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
            if files:
                letters_classes.append(label)
                for fpath in files:
                    arr = np.load(fpath)
                    sequences.append(arr)
                    labels.append(label)
        print(f"[INFO] Dimuat {len(sequences)} sampel huruf dari {len(letters_classes)} kelas di {letters_dir}")
    else:
        print(f"[WARN] Folder huruf tidak ditemukan: {letters_dir}")

    count_letters = len(sequences)

    # 2. Muat Kata
    if os.path.isdir(words_dir):
        word_folders = sorted([
            d for d in os.listdir(words_dir)
            if os.path.isdir(os.path.join(words_dir, d))
        ])
        for label in word_folders:
            folder_path = os.path.join(words_dir, label)
            files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
            if files:
                words_classes.append(label)
                for fpath in files:
                    arr = np.load(fpath)
                    sequences.append(arr)
                    labels.append(label)
        print(f"[INFO] Dimuat {len(sequences) - count_letters} sampel kata dari {len(words_classes)} kelas di {words_dir}")
    else:
        print(f"[WARN] Folder kata tidak ditemukan: {words_dir}")

    return sequences, labels, letters_classes, words_classes


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess gabungan 36 kelas (Huruf + Kata)")
    parser.add_argument("--config", default="configs/combined_90.yaml", help="Path config gabungan")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config tidak ditemukan: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    prep_cfg = config["preprocessing"]

    random_seed = prep_cfg.get("random_seed", 42)
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("\n=======================================================")
    print("PREPROCESSING GABUNGAN SIGN LANGUAGE (HURUF & KATA)")
    print("=======================================================")

    sequences, labels, letters_classes, words_classes = load_raw_samples(
        paths["letters_landmarks"],
        paths["words_landmarks"],
    )

    if not sequences:
        print("[ERROR] Tidak ada sampel yang berhasil dimuat!")
        return 1

    all_classes = sorted(list(set(labels)))
    num_classes = len(all_classes)
    label_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes)}
    idx_to_label = {str(i): cls_name for i, cls_name in enumerate(all_classes)}

    print(f"\n[INFO] Total Kelas: {num_classes} ({len(letters_classes)} Huruf + {len(words_classes)} Kata)")
    print(f"[INFO] Total Sampel: {len(sequences)}")

    # Ekstraksi fitur 140-dimensi
    print("[INFO] Mengekstrak 140 fitur motion-aware untuk setiap sequence...")
    features_list = [extract_combined_features(seq) for seq in sequences]

    int_labels = np.array([label_to_idx[lbl] for lbl in labels], dtype=np.int64)

    test_size = prep_cfg.get("test_size", 0.15)
    val_size = prep_cfg.get("val_size", 0.15)
    val_relative_size = val_size / (1.0 - test_size)

    indices = np.arange(len(features_list))

    # Split train+val vs test
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        indices,
        int_labels,
        test_size=test_size,
        stratify=int_labels,
        random_state=random_seed,
    )

    # Split train vs val
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_val_idx,
        y_train_val,
        test_size=val_relative_size,
        stratify=y_train_val,
        random_state=random_seed,
    )

    max_seq_len = prep_cfg.get("max_seq_length", 90)

    # Pad data
    train_feats = [features_list[i] for i in train_idx]
    val_feats = [features_list[i] for i in val_idx]
    test_feats = [features_list[i] for i in test_idx]

    x_train, len_train = pad_features(train_feats, max_seq_len)
    x_val, len_val = pad_features(val_feats, max_seq_len)
    x_test, len_test = pad_features(test_feats, max_seq_len)

    out_dir = paths["processed"]
    os.makedirs(out_dir, exist_ok=True)

    # Simpan NPZ
    np.savez_compressed(
        os.path.join(out_dir, "train.npz"),
        data=x_train,
        lengths=len_train,
        labels=y_train,
    )
    np.savez_compressed(
        os.path.join(out_dir, "val.npz"),
        data=x_val,
        lengths=len_val,
        labels=y_val,
    )
    np.savez_compressed(
        os.path.join(out_dir, "test.npz"),
        data=x_test,
        lengths=len_test,
        labels=y_test,
    )

    # Simpan label encoder
    encoder_meta = {
        "num_classes": num_classes,
        "classes": all_classes,
        "letters_classes": letters_classes,
        "words_classes": words_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }
    with open(os.path.join(out_dir, "label_encoder.json"), "w", encoding="utf-8") as f:
        json.dump(encoder_meta, f, indent=2)

    # Simpan summary
    summary = {
        "num_classes": num_classes,
        "total_samples": len(sequences),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "feature_dim": COMBINED_INPUT_SIZE,
        "max_seq_length": max_seq_len,
    }
    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("HASIL PREPROCESSING GABUNGAN:")
    print(f"  Total Data : {len(sequences)} sampel")
    print(f"  Train Set  : {len(train_idx)} sampel ({x_train.shape})")
    print(f"  Val Set    : {len(val_idx)} sampel ({x_val.shape})")
    print(f"  Test Set   : {len(test_idx)} sampel ({x_test.shape})")
    print(f"  Output Dir : {out_dir}")
    print("=======================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

