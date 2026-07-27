"""Preprocessing khusus kata dengan fitur bentuk dan gerakan tangan.

Pipeline huruf tetap menggunakan ``preprocess.py``. Script ini membaca landmark
mentah berbentuk (frame, 42, 3), lalu membangun 140 fitur per frame:

* 126 fitur bentuk tangan wrist-relative (2 x 21 x 3)
* 6 koordinat wrist global (2 x xyz)
* 6 perpindahan wrist dari posisi valid pertama (2 x xyz)
* 2 mask kehadiran tangan

Panjang sequence asli disimpan agar packed BiLSTM dapat mengabaikan padding.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES = 3
LOCAL_SHAPE_SIZE = LANDMARKS_PER_HAND * COORDINATES
FEATURES_PER_HAND = LOCAL_SHAPE_SIZE + 3 + 3 + 1
WORD_INPUT_SIZE = NUM_HANDS * FEATURES_PER_HAND  # 140


def load_landmarks(input_dir: str):
    """Load landmark, label, dan path sampel secara deterministik."""
    root = Path(input_dir)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"Tidak ada folder kelas di {input_dir}")

    label_to_idx = {name: idx for idx, name in enumerate(classes)}
    sequences, labels, sample_paths = [], [], []

    for class_name in classes:
        files = sorted((root / class_name).glob("*.npy"))
        for file_path in tqdm(files, desc=f"Load {class_name}"):
            sequence = np.load(file_path)
            expected_shape = (NUM_HANDS * LANDMARKS_PER_HAND, COORDINATES)
            if sequence.ndim != 3 or sequence.shape[1:] != expected_shape:
                raise ValueError(
                    f"Shape {file_path} adalah {sequence.shape}; "
                    f"diharapkan (frames, {expected_shape[0]}, {expected_shape[1]}). "
                    "Ekstrak landmark kata dengan --max_num_hands 2."
                )
            if sequence.shape[0] == 0:
                raise ValueError(f"Sequence kosong: {file_path}")
            sequences.append(sequence.astype(np.float32, copy=False))
            labels.append(label_to_idx[class_name])
            sample_paths.append(str(file_path.as_posix()))

    return sequences, labels, sample_paths, label_to_idx


def hand_presence(sequence: np.ndarray) -> np.ndarray:
    """Return mask (frames, 2) untuk tangan yang benar-benar terdeteksi."""
    masks = []
    for hand_idx in range(NUM_HANDS):
        start = hand_idx * LANDMARKS_PER_HAND
        block = sequence[:, start:start + LANDMARKS_PER_HAND, :]
        masks.append(np.any(np.abs(block) > 1e-8, axis=(1, 2)))
    return np.stack(masks, axis=1)


def extract_word_features(sequence: np.ndarray) -> np.ndarray:
    """Bangun fitur bentuk lokal, posisi wrist, displacement, dan presence."""
    presence = hand_presence(sequence)
    hand_features = []

    for hand_idx in range(NUM_HANDS):
        start = hand_idx * LANDMARKS_PER_HAND
        block = sequence[:, start:start + LANDMARKS_PER_HAND, :]
        valid = presence[:, hand_idx]

        wrist = block[:, 0, :]
        centered = block - wrist[:, None, :]
        palm_scale = np.linalg.norm(centered[:, 9, :], axis=1)
        safe_scale = np.where((palm_scale > 1e-6) & valid, palm_scale, 1.0)
        local_shape = centered / safe_scale[:, None, None]

        # Missing-hand frames harus tetap nol, bukan menjadi fitur sintetis.
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
    if result.shape[1] != WORD_INPUT_SIZE:
        raise RuntimeError(f"Ukuran fitur {result.shape[1]} != {WORD_INPUT_SIZE}")
    return result


def augment_landmarks(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentasi ringan yang tidak mengubah slot tangan kosong menjadi noise."""
    old_length = sequence.shape[0]
    speed_factor = rng.uniform(0.90, 1.10)
    new_length = max(2, int(round(old_length * speed_factor)))
    old_time = np.arange(old_length, dtype=np.float32)
    new_time = np.linspace(0, old_length - 1, new_length, dtype=np.float32)

    warped = np.empty((new_length, sequence.shape[1], COORDINATES), dtype=np.float32)
    for landmark_idx in range(sequence.shape[1]):
        for coordinate_idx in range(COORDINATES):
            warped[:, landmark_idx, coordinate_idx] = np.interp(
                new_time,
                old_time,
                sequence[:, landmark_idx, coordinate_idx],
            )

    original_presence = hand_presence(sequence)
    nearest_indices = np.clip(np.rint(new_time).astype(int), 0, old_length - 1)
    warped_presence = original_presence[nearest_indices]

    angle = np.deg2rad(rng.uniform(-8.0, 8.0))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    spatial_scale = rng.uniform(0.95, 1.05)
    translation = rng.uniform(-0.02, 0.02, size=2).astype(np.float32)

    for hand_idx in range(NUM_HANDS):
        start = hand_idx * LANDMARKS_PER_HAND
        end = start + LANDMARKS_PER_HAND
        valid = warped_presence[:, hand_idx]
        block = warped[:, start:end, :]

        if np.any(valid):
            xy = block[valid, :, :2]
            xy = ((xy - 0.5) @ rotation.T) * spatial_scale + 0.5 + translation
            block[valid, :, :2] = xy
            noise = rng.normal(0.0, 0.003, size=block[valid].shape).astype(np.float32)
            block[valid] += noise

        block[~valid] = 0.0
        warped[:, start:end, :] = block

    return warped


def split_indices(labels, test_size: float, val_size: float, random_seed: int):
    labels_array = np.asarray(labels)
    all_indices = np.arange(len(labels_array))
    train_val, test = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels_array,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=labels_array[train_val],
    )
    return {"train": train, "val": val, "test": test}


def pad_features(feature_sequences, max_seq_length: int):
    data = np.zeros(
        (len(feature_sequences), max_seq_length, WORD_INPUT_SIZE),
        dtype=np.float32,
    )
    lengths = np.zeros(len(feature_sequences), dtype=np.int64)
    for idx, features in enumerate(feature_sequences):
        valid_length = min(features.shape[0], max_seq_length)
        data[idx, :valid_length] = features[:valid_length]
        lengths[idx] = valid_length
    return data, lengths


def save_tensor(array, path: Path, dtype):
    tensor = torch.as_tensor(array, dtype=dtype)
    torch.save(tensor, path)
    print(f"  [SAVED] {path} - shape: {tuple(tensor.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocessing motion-aware khusus kata")
    parser.add_argument("--config", default="configs/words_motion.yaml")
    parser.add_argument("--augment_factor", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    preprocessing = config["preprocessing"]
    max_seq_length = args.max_seq_length or preprocessing["max_seq_length"]
    augment_factor = (
        preprocessing["augment_factor"]
        if args.augment_factor is None
        else args.augment_factor
    )
    if augment_factor < 0:
        raise ValueError("augment_factor tidak boleh negatif")

    sequences, labels, sample_paths, label_to_idx = load_landmarks(
        config["paths"]["landmarks"]
    )
    indices = split_indices(
        labels,
        preprocessing["test_size"],
        preprocessing["val_size"],
        preprocessing["random_seed"],
    )
    rng = np.random.default_rng(preprocessing["random_seed"])
    output_dir = Path(config["paths"]["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Motion-aware word preprocessing")
    print(f"  Samples          : {len(sequences)}")
    print(f"  Classes          : {len(label_to_idx)}")
    print(f"  Feature size     : {WORD_INPUT_SIZE}")
    print(f"  Max seq length   : {max_seq_length}")
    print(f"  Augment factor   : {augment_factor}")

    split_manifest = {}
    for split_name in ("train", "val", "test"):
        split_idx = indices[split_name]
        split_sequences = [sequences[idx] for idx in split_idx]
        split_labels = [labels[idx] for idx in split_idx]
        split_manifest[split_name] = [sample_paths[idx] for idx in split_idx]

        if split_name == "train" and augment_factor:
            original_sequences = list(split_sequences)
            original_labels = list(split_labels)
            for sequence, label in zip(original_sequences, original_labels):
                for _ in range(augment_factor):
                    split_sequences.append(augment_landmarks(sequence, rng))
                    split_labels.append(label)

        features = [extract_word_features(sequence) for sequence in split_sequences]
        data, lengths = pad_features(features, max_seq_length)

        save_tensor(data, output_dir / f"{split_name}_data.pt", torch.float32)
        save_tensor(np.asarray(split_labels), output_dir / f"{split_name}_labels.pt", torch.long)
        save_tensor(lengths, output_dir / f"{split_name}_lengths.pt", torch.long)

    idx_to_label = {str(index): label for label, index in label_to_idx.items()}
    with open(output_dir / "label_encoder.json", "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "num_classes": len(label_to_idx),
            },
            output_file,
            indent=2,
        )

    with open(output_dir / "feature_metadata.json", "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "pipeline": "words_motion_v1",
                "input_size": WORD_INPUT_SIZE,
                "max_seq_length": max_seq_length,
                "augment_factor": augment_factor,
                "features_per_hand": {
                    "local_wrist_relative_shape": LOCAL_SHAPE_SIZE,
                    "global_wrist_position": 3,
                    "wrist_displacement": 3,
                    "presence_mask": 1,
                },
            },
            output_file,
            indent=2,
        )

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as output_file:
        json.dump(split_manifest, output_file, indent=2)

    print(f"\n[DONE] Processed word data tersimpan di {output_dir}")


if __name__ == "__main__":
    main()
