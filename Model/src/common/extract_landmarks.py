"""
=============================================================
Tahap 1: Ekstraksi Landmark dari Video Dataset
=============================================================
Mengekstrak titik landmark tangan menggunakan MediaPipe Hands
dari setiap frame video. Output berupa file .npy per video.

Input:  dataset/{task}/raw/{nama_kelas}/video.mp4
Output: dataset/{task}/landmarks/{nama_kelas}/video.npy
        Shape: (num_frames, 21, 3) jika max_num_hands=1 (huruf)
        Shape: (num_frames, 42, 3) jika max_num_hands=2 (kata)
               — baris 0-20 = tangan kiri, baris 21-41 = tangan kanan
               (slot tetap berdasarkan handedness, diisi nol jika
               tangan tersebut tidak terdeteksi pada frame tertentu)
=============================================================
"""

import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


def extract_landmarks_from_video(video_path: str, hands, max_num_hands: int = 1) -> np.ndarray:
    """
    Ekstrak landmark tangan dari satu video.

    Args:
        video_path: Path ke file video
        hands: Instance MediaPipe Hands
        max_num_hands: 1 (huruf, 1 tangan) atau 2 (kata, bisa 2 tangan).
                       Saat 2, tiap tangan ditaruh di slot tetap berdasarkan
                       handedness ([Left, Right]) supaya konsisten antar frame.

    Returns:
        np.ndarray: Array landmark, shape (num_frames, 21*max_num_hands, 3).
                    Mengembalikan array kosong jika video tidak punya frame.
    """
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi BGR ke RGB (MediaPipe membutuhkan RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if max_num_hands == 1:
            if results.multi_hand_landmarks:
                # Ambil tangan pertama yang terdeteksi
                hand_landmarks = results.multi_hand_landmarks[0]
                frame_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            else:
                # Jika tidak ada tangan terdeteksi, isi dengan zeros
                frame_landmarks = np.zeros((21, 3)).tolist()
        else:
            # Slot tetap [Left, Right] berdasarkan handedness MediaPipe,
            # supaya urutan tangan konsisten antar frame & antar video.
            slots = {"Left": np.zeros((21, 3)), "Right": np.zeros((21, 3))}
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label
                    slots[label] = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    )
            frame_landmarks = np.vstack([slots["Left"], slots["Right"]]).tolist()

        landmarks_sequence.append(frame_landmarks)

    cap.release()

    if len(landmarks_sequence) == 0:
        return np.array([])

    return np.array(landmarks_sequence, dtype=np.float32)


def process_dataset(
    input_dir: str,
    output_dir: str,
    overwrite: bool = False,
    max_num_hands: int = 1
) -> None:
    """
    Proses seluruh dataset video dan ekstrak landmark.

    Args:
        input_dir: Path ke folder dataset/{task}/raw/
        output_dir: Path ke folder output dataset/{task}/landmarks/
        overwrite: Jika False (default), video yang sudah punya file .npy di
                   output_dir akan dilewati (skip) — hanya video baru yang diproses.
                   Jika True, semua video diekstrak ulang.
        max_num_hands: 1 untuk huruf, 2 untuk kata (lihat extract_landmarks_from_video)
    """
    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Dapatkan semua kelas (subfolder)
    classes = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    if not classes:
        print(f"[ERROR] Tidak ada subfolder kelas ditemukan di: {input_dir}")
        return

    print(f"[INFO] Ditemukan {len(classes)} kelas: {classes}")

    total_videos = 0
    total_success = 0
    total_empty = 0
    total_skipped = 0

    for class_name in classes:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Dapatkan semua file video
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        video_files = sorted([
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(video_extensions)
        ])

        print(f"\n[INFO] Kelas '{class_name}': {len(video_files)} video")

        for video_file in tqdm(video_files, desc=f"  {class_name}"):
            total_videos += 1
            video_path = os.path.join(class_input_dir, video_file)

            # Skip video yang sudah pernah diekstrak sebelumnya
            output_filename = os.path.splitext(video_file)[0] + ".npy"
            output_path = os.path.join(class_output_dir, output_filename)
            if not overwrite and os.path.exists(output_path):
                total_skipped += 1
                continue

            # Ekstrak landmark
            landmarks = extract_landmarks_from_video(video_path, hands, max_num_hands=max_num_hands)

            if landmarks.size == 0:
                print(f"  [WARN] Tidak ada landmark terdeteksi: {video_file}")
                total_empty += 1
                continue

            # Simpan sebagai .npy
            np.save(output_path, landmarks)
            total_success += 1

    hands.close()

    print(f"\n{'='*50}")
    print(f"[DONE] Ekstraksi Landmark Selesai")
    print(f"  Total video    : {total_videos}")
    print(f"  Berhasil       : {total_success}")
    print(f"  Dilewati (skip): {total_skipped}")
    print(f"  Kosong/gagal   : {total_empty}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi landmark tangan dari video dataset menggunakan MediaPipe"
    )
    parser.add_argument(
        "--input_dir", type=str, default="dataset/letters/raw",
        help="Path ke folder dataset video mentah (default: dataset/letters/raw)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/letters/landmarks",
        help="Path ke folder output landmark (default: dataset/letters/landmarks)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ekstrak ulang semua video meski .npy sudah ada (default: skip video yang sudah diekstrak)"
    )
    parser.add_argument(
        "--max_num_hands", type=int, default=1, choices=[1, 2],
        help="Jumlah tangan yang dideteksi: 1 untuk huruf, 2 untuk kata (default: 1)"
    )
    args = parser.parse_args()

    print("[INFO] Memulai ekstraksi landmark...")
    print(f"  Input         : {args.input_dir}")
    print(f"  Output        : {args.output_dir}")
    print(f"  Overwrite     : {args.overwrite}")
    print(f"  Max num hands : {args.max_num_hands}")

    process_dataset(
        args.input_dir, args.output_dir,
        overwrite=args.overwrite, max_num_hands=args.max_num_hands
    )


if __name__ == "__main__":
    main()
