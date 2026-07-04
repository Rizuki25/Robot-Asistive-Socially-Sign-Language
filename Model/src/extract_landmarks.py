"""
=============================================================
Tahap 1: Ekstraksi Landmark dari Video Dataset
=============================================================
Mengekstrak 21 titik landmark tangan menggunakan MediaPipe Hands
dari setiap frame video. Output berupa file .npy per video.

Input:  dataset/raw/{nama_kelas}/video.mp4
Output: dataset/landmarks/{nama_kelas}/video.npy
        Shape: (num_frames, 21, 3)
=============================================================
"""

import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


def extract_landmarks_from_video(video_path: str, hands) -> np.ndarray:
    """
    Ekstrak landmark tangan dari satu video.

    Args:
        video_path: Path ke file video
        hands: Instance MediaPipe Hands

    Returns:
        np.ndarray: Array landmark dengan shape (num_frames, 21, 3)
                    Mengembalikan array kosong jika tidak ada tangan terdeteksi
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

        if results.multi_hand_landmarks:
            # Ambil tangan pertama yang terdeteksi
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []

            for landmark in hand_landmarks.landmark:
                frame_landmarks.append([landmark.x, landmark.y, landmark.z])

            landmarks_sequence.append(frame_landmarks)
        else:
            # Jika tidak ada tangan terdeteksi, isi dengan zeros
            landmarks_sequence.append(np.zeros((21, 3)).tolist())

    cap.release()

    if len(landmarks_sequence) == 0:
        return np.array([])

    return np.array(landmarks_sequence, dtype=np.float32)


def process_dataset(input_dir: str, output_dir: str) -> None:
    """
    Proses seluruh dataset video dan ekstrak landmark.

    Args:
        input_dir: Path ke folder dataset/raw/
        output_dir: Path ke folder output dataset/landmarks/
    """
    # Inisialisasi MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
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

            # Ekstrak landmark
            landmarks = extract_landmarks_from_video(video_path, hands)

            if landmarks.size == 0:
                print(f"  [WARN] Tidak ada landmark terdeteksi: {video_file}")
                total_empty += 1
                continue

            # Simpan sebagai .npy
            output_filename = os.path.splitext(video_file)[0] + ".npy"
            output_path = os.path.join(class_output_dir, output_filename)
            np.save(output_path, landmarks)
            total_success += 1

    hands.close()

    print(f"\n{'='*50}")
    print(f"[DONE] Ekstraksi Landmark Selesai")
    print(f"  Total video    : {total_videos}")
    print(f"  Berhasil       : {total_success}")
    print(f"  Kosong/gagal   : {total_empty}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi landmark tangan dari video dataset menggunakan MediaPipe"
    )
    parser.add_argument(
        "--input_dir", type=str, default="dataset/raw",
        help="Path ke folder dataset video mentah (default: dataset/raw)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="dataset/landmarks",
        help="Path ke folder output landmark (default: dataset/landmarks)"
    )
    args = parser.parse_args()

    print("[INFO] Memulai ekstraksi landmark...")
    print(f"  Input  : {args.input_dir}")
    print(f"  Output : {args.output_dir}")

    process_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
