"""
Inference Video untuk Model Gabungan 36 Kelas (Huruf & Kata).
============================================================
Fitur Realtime-Like Display (Sama persis seperti Webcam):
- Frame 0 s/d 44: Status "Merekam Gerakan..." (Mengumpulkan frame: X/45).
- Frame 45+: Menganalisis kandidat secara berkala.
- Akhir Gerakan: Prediksi Terkunci (Warna Hijau & Audio Bunyi).
- Pilihan kategori interaktif (Huruf, Kata, Semua, atau Kelas Spesifik).
- Tombol: [SPACE] Putar Ulang, [R] Random Video Lain, [Q/Esc] Keluar.

Jalankan dari folder Model/:
    python -m src.combined.predict_video
"""

import argparse
import glob
import json
import os
import random
import time
from collections import deque
from typing import Optional, Tuple

try:
    import winsound
except ImportError:
    winsound = None

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from src.common.utils import get_device
from src.combined.model import CombinedBiLSTM
from src.combined.preprocess import extract_combined_features, pad_features

WINDOW_NAME = "Prediksi Video (36 Kelas) - SPACE: Ulangi | R: Video Lain | Q: Keluar"
MODEL_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def resolve_input_path(path: str) -> str:
    """Resolve path input dari direktori kerja atau root Model/."""
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(MODEL_ROOT, path)


def load_label_encoder(path: str) -> Tuple[int, list]:
    """Load class mapping index."""
    resolved = resolve_input_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        encoder = json.load(f)
    num_classes = int(encoder["num_classes"])
    idx_to_label = encoder["idx_to_label"]
    class_names = [idx_to_label[str(idx)] for idx in range(num_classes)]
    return num_classes, class_names


def draw_landmarks(frame: np.ndarray, frame_landmarks: np.ndarray) -> None:
    """Gambar titik landmark dan koneksi tangan."""
    height, width = frame.shape[:2]
    num_hands = frame_landmarks.shape[0] // 21
    colors = [(0, 255, 255), (255, 140, 0)]

    for hand_index in range(num_hands):
        hand = frame_landmarks[hand_index * 21 : (hand_index + 1) * 21]
        if np.allclose(hand, 0):
            continue

        points = [
            (int(np.clip(landmark[0], 0, 1) * width), int(np.clip(landmark[1], 0, 1) * height))
            for landmark in hand
        ]
        color = colors[hand_index % len(colors)]
        for start, end in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 5, (255, 255, 255), 1, cv2.LINE_AA)


def extract_landmarks_from_video(
    video_path: str,
    max_num_hands: int = 2,
) -> Tuple[np.ndarray, list, float, int, int]:
    """Ekstrak semua frame dan landmark MediaPipe dari file video."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Video tidak dapat dibuka: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    frames = []
    landmarks_list = []

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frames.append(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            slots = {
                "Left": np.zeros((21, 3), dtype=np.float32),
                "Right": np.zeros((21, 3), dtype=np.float32),
            }
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label in slots:
                        slots[label] = np.asarray(
                            [[point.x, point.y, point.z] for point in hand.landmark],
                            dtype=np.float32,
                        )
            landmarks_list.append(np.vstack([slots["Left"], slots["Right"]]))
    finally:
        capture.release()
        mp_hands.close()

    return np.asarray(landmarks_list, dtype=np.float32), frames, fps, width, height


def predict_sequence(
    landmarks: np.ndarray,
    config: dict,
    model: CombinedBiLSTM,
    device: torch.device,
) -> Tuple[str, float]:
    """Prediksi label 36 kelas dari array landmark video."""
    max_seq_length = int(config["preprocessing"]["max_seq_length"])
    if len(landmarks) > max_seq_length:
        sample_indices = np.rint(
            np.linspace(0, len(landmarks) - 1, max_seq_length)
        ).astype(np.int64)
        landmarks = landmarks[sample_indices]

    features = extract_combined_features(landmarks)
    data, lengths = pad_features([features], max_seq_length)
    inputs = torch.from_numpy(data).to(device)
    lengths_t = torch.from_numpy(lengths).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(inputs, lengths_t), dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return str(int(pred.item())), float(conf.item())


class CombinedAudioPlayer:
    """Player audio terpadu huruf dan kata."""

    def __init__(self, letters_dir: str = "assets/letters_audio", words_dir: str = "assets/words_audio"):
        self.letters_dir = resolve_input_path(letters_dir)
        self.words_dir = resolve_input_path(words_dir)
        self.available = winsound is not None

    def play(self, label: str) -> None:
        if not self.available:
            return
        candidates = [
            os.path.join(self.words_dir, f"{label}.wav"),
            os.path.join(self.words_dir, f"{label.lower()}.wav"),
            os.path.join(self.letters_dir, f"{label.upper()}.wav"),
            os.path.join(self.letters_dir, f"{label}.wav"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
                    return
                except Exception:
                    pass


def get_available_videos() -> Tuple[dict, dict]:
    """Cari semua video yang tersedia di folder dataset letters dan words."""
    letters_videos = {}
    words_videos = {}

    for letter_dir in ["dataset/letters/raw_90", "dataset/letters/raw"]:
        abs_l = resolve_input_path(letter_dir)
        if os.path.isdir(abs_l):
            for cls_folder in os.listdir(abs_l):
                folder_path = os.path.join(abs_l, cls_folder)
                if os.path.isdir(folder_path):
                    files = glob.glob(os.path.join(folder_path, "*.*"))
                    vfiles = [f for f in files if f.lower().endswith((".avi", ".mp4", ".mov"))]
                    if vfiles:
                        letters_videos.setdefault(cls_folder, []).extend(vfiles)
            if letters_videos:
                break

    for word_dir in ["dataset/words/raw_90", "dataset/words/raw"]:
        abs_w = resolve_input_path(word_dir)
        if os.path.isdir(abs_w):
            for cls_folder in os.listdir(abs_w):
                folder_path = os.path.join(abs_w, cls_folder)
                if os.path.isdir(folder_path):
                    files = glob.glob(os.path.join(folder_path, "*.*"))
                    vfiles = [f for f in files if f.lower().endswith((".avi", ".mp4", ".mov"))]
                    if vfiles:
                        words_videos.setdefault(cls_folder, []).extend(vfiles)
            if words_videos:
                break

    return letters_videos, words_videos


def select_video_interactive(category: Optional[str], class_name: Optional[str]) -> Tuple[str, str]:
    """Pilih video secara interaktif atau via argumen."""
    letters_videos, words_videos = get_available_videos()
    all_videos = {}
    for k, v in letters_videos.items():
        all_videos.setdefault(k, []).extend(v)
    for k, v in words_videos.items():
        all_videos.setdefault(k, []).extend(v)

    if class_name:
        for k in all_videos:
            if k.lower() == class_name.strip().lower():
                return random.choice(all_videos[k]), k
        print(f"[WARN] Kelas '{class_name}' tidak ditemukan di dataset. Membuka menu...")

    if category:
        cat_lower = category.strip().lower()
        if cat_lower in ["1", "huruf", "letter", "letters"]:
            chosen_cls = random.choice(list(letters_videos.keys()))
            return random.choice(letters_videos[chosen_cls]), chosen_cls
        elif cat_lower in ["2", "kata", "word", "words"]:
            chosen_cls = random.choice(list(words_videos.keys()))
            return random.choice(words_videos[chosen_cls]), chosen_cls
        elif cat_lower in ["3", "semua", "all", "combined"]:
            chosen_cls = random.choice(list(all_videos.keys()))
            return random.choice(all_videos[chosen_cls]), chosen_cls

    print("\n" + "=" * 60)
    print("PILIH SUMBER VIDEO UNTUK PENGUJIAN MODEL GABUNGAN (36 KELAS)")
    print("=" * 60)
    print(f" [1] Random Video HURUF (Tersedia: {len(letters_videos)} kelas huruf A-Z)")
    print(f" [2] Random Video KATA  (Tersedia: {len(words_videos)} kelas kata)")
    print(f" [3] Random Video SEMUA (Tersedia: {len(all_videos)} total kelas)")
    print(" [4] Ketik Nama Kelas Spesifik (misal: Makan, Apa, Halo, P, D, dll.)")
    print(" [Q] Keluar")
    print("=" * 60)

    while True:
        try:
            choice = input("Pilihan Anda (1/2/3/4 atau Nama Kelas): ").strip()
        except (KeyboardInterrupt, EOFError):
            return "", ""

        if choice.lower() in ["q", "exit", "keluar"]:
            return "", ""
        elif choice == "1" or choice.lower() in ["huruf", "letter"]:
            if not letters_videos:
                print("[ERROR] Tidak ada video huruf ditemukan!")
                continue
            chosen_cls = random.choice(list(letters_videos.keys()))
            return random.choice(letters_videos[chosen_cls]), chosen_cls
        elif choice == "2" or choice.lower() in ["kata", "word"]:
            if not words_videos:
                print("[ERROR] Tidak ada video kata ditemukan!")
                continue
            chosen_cls = random.choice(list(words_videos.keys()))
            return random.choice(words_videos[chosen_cls]), chosen_cls
        elif choice == "3" or choice.lower() in ["semua", "all"]:
            chosen_cls = random.choice(list(all_videos.keys()))
            return random.choice(all_videos[chosen_cls]), chosen_cls
        else:
            target_name = choice
            if choice == "4":
                target_name = input("Masukkan nama kelas: ").strip()
            for k in all_videos:
                if k.lower() == target_name.lower():
                    return random.choice(all_videos[k]), k
            print(f"[ERROR] Kelas '{target_name}' tidak ditemukan. Coba lagi...")


def play_and_display_video(
    video_path: str,
    true_label: str,
    predicted_label: str,
    confidence: float,
    landmarks: np.ndarray,
    frames: list,
    fps: float,
    config: dict,
    model: CombinedBiLSTM,
    class_names: list,
    device: torch.device,
    min_recording_frames: int = 45,
    audio_player: Optional[CombinedAudioPlayer] = None,
) -> str:
    """
    Putar video dengan alur display bertahap (sama seperti realtime webcam):
    - Frame 0 s/d 44: 'Merekam Gerakan...' (Mengumpulkan frame: X/45)
    - Frame 45+: 'Menganalisis Gerakan / Kandidat...'
    - Selesai/Kunci: 'Prediksi: [Label]' (Terkunci & Audio Bunyi).
    """
    if not frames:
        return "next"

    h, w = frames[0].shape[:2]
    display_w = 640
    display_h = int(h * (display_w / w))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_w, display_h)

    is_correct = (true_label.lower() == predicted_label.lower()) if true_label else None
    status_text = "TEPAT" if is_correct else "BEDA"
    status_color = (0, 255, 0) if is_correct else (0, 165, 255)

    delay_ms = max(1, int(1000 / (fps or 30.0)))
    total_frames = len(frames)

    # Tentukan frame kunci hasil (misal frame ke-55 atau 15 frame sebelum akhir video)
    lock_frame = max(min_recording_frames + 5, min(total_frames - 10, 60))

    while True:
        candidate_label = ""
        candidate_conf = 0.0
        audio_played = False

        for idx in range(total_frames):
            frame = frames[idx].copy()
            if idx < len(landmarks):
                draw_landmarks(frame, landmarks[idx])

            if (frame.shape[1], frame.shape[0]) != (display_w, display_h):
                frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            # Header overlay 70px (sama seperti webcam)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_w, 72), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Baris 1: Filename & Label Asli
            fname = os.path.basename(video_path)
            cv2.putText(frame, f"File: {fname}", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            if true_label:
                cv2.putText(frame, f"Label Asli: {true_label}", (display_w - 180, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

            # Logika Status Bertahap Sesuai Frame Saat Ini
            if idx + 1 < min_recording_frames:
                title = "Merekam Gerakan..."
                detail = f"Mengumpulkan frame: {idx + 1}/{min_recording_frames}"
                title_color = (0, 215, 255)
            elif idx + 1 < lock_frame:
                # Lakukan inferensi kandidat berkala setiap 3 frame
                if (idx % 3 == 0) and idx < len(landmarks):
                    c_idx_str, c_conf = predict_sequence(landmarks[: idx + 1], config, model, device)
                    candidate_label = class_names[int(c_idx_str)]
                    candidate_conf = c_conf

                if candidate_label:
                    title = f"Kandidat: {candidate_label}"
                    detail = f"Conf: {candidate_conf * 100:.1f}% | Menganalisis gerakan..."
                else:
                    title = "Menganalisis Gerakan..."
                    detail = "Menunggu konsensus prediksi"
                title_color = (0, 215, 255)
            else:
                # Hasil Terkunci (Gerakan Selesai)
                title = f"Prediksi: {predicted_label}"
                detail = f"Confidence: {confidence * 100:.1f}% | Terkunci"
                title_color = (0, 255, 0) if is_correct else (0, 215, 255)

                if is_correct is not None:
                    cv2.putText(frame, f"[{status_text}]", (display_w - 110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

                # Bunyikan audio tepat saat hasil terkunci
                if not audio_played and audio_player is not None:
                    audio_player.play(predicted_label)
                    audio_played = True

            # Baris 2: Judul Status / Hasil Prediksi
            cv2.putText(frame, title, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, title_color, 2, cv2.LINE_AA)

            # Baris 3: Detail & Petunjuk Tombol
            cv2.putText(frame, detail, (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(frame, "[SPACE]: Ulangi | [R]: Video Lain | [Q]: Keluar", (display_w - 290, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            elif key in (ord("r"), ord("R")):
                return "random"
            elif key == ord(" "):
                break  # Replay loop

        # Setelah video selesai diputar, tahan di frame terakhir
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            elif key in (ord("r"), ord("R")):
                return "random"
            elif key == ord(" "):
                break  # Replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict single/random video with 36-Class Combined Model")
    parser.add_argument("--video_path", default=None, help="Path ke file video input spesifik")
    parser.add_argument("--category", default=None, choices=["huruf", "kata", "semua"], help="Pilih kategori video (huruf/kata/semua)")
    parser.add_argument("--class_name", default=None, help="Pilih kelas video spesifik (misal: Makan, Halo, P, dll.)")
    parser.add_argument("--config", default="configs/combined_90.yaml", help="Path config gabungan")
    parser.add_argument("--model_path", default=None, help="Path checkpoint model (default: best_model.pth)")
    parser.add_argument("--min_recording_frames", type=int, default=45, help="Frame minimum sebelum menebak (default: 45)")
    parser.add_argument("--no_speech", action="store_true", help="Nonaktifkan pemutaran audio")
    parser.add_argument("--no_display", action="store_true", help="Jalankan headless tanpa tampilan window")
    args = parser.parse_args()

    config_path = resolve_input_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()
    paths = config["paths"]
    encoder_path = resolve_input_path(os.path.join(paths["processed"], "label_encoder.json"))
    num_classes, class_names = load_label_encoder(encoder_path)

    model_path = args.model_path or os.path.join(paths["models"], "best_model.pth")
    model_path = resolve_input_path(model_path)

    if not os.path.isfile(model_path):
        print(f"[ERROR] Model checkpoint tidak ditemukan: {model_path}")
        return 1

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

    audio_player = None
    if not args.no_speech:
        audio_player = CombinedAudioPlayer()

    current_video_path = args.video_path
    current_true_label = args.class_name or ""

    try:
        while True:
            if not current_video_path:
                vpath, tlabel = select_video_interactive(args.category, args.class_name)
                if not vpath:
                    break
                current_video_path = vpath
                current_true_label = tlabel

            video_path = resolve_input_path(current_video_path)
            if not os.path.isfile(video_path):
                print(f"[ERROR] Video tidak ditemukan: {video_path}")
                break

            print(f"\n[INFO] Mengekstrak landmark dan menganalisis video: {video_path}")
            landmarks, frames, fps, width, height = extract_landmarks_from_video(video_path)
            label_idx_str, confidence = predict_sequence(landmarks, config, model, device)
            predicted_label = class_names[int(label_idx_str)]

            print("=" * 60)
            print(f"HASIL PREDIKSI VIDEO (36 KELAS):")
            print(f"  File Video     : {os.path.basename(video_path)}")
            if current_true_label:
                print(f"  Label Asli     : {current_true_label}")
            print(f"  Prediksi Model : {predicted_label}")
            print(f"  Confidence     : {confidence * 100:.2f}%")
            if current_true_label:
                status = "TEPAT [BENAR]" if current_true_label.lower() == predicted_label.lower() else "BEDA [SALAH]"
                print(f"  Status         : {status}")
            print("=" * 60 + "\n")

            if not args.no_display:
                action = play_and_display_video(
                    video_path,
                    current_true_label,
                    predicted_label,
                    confidence,
                    landmarks,
                    frames,
                    fps,
                    config,
                    model,
                    class_names,
                    device,
                    min_recording_frames=args.min_recording_frames,
                    audio_player=audio_player,
                )
                if action == "quit":
                    break
                elif action == "random":
                    current_video_path = None
                    continue
            else:
                break

            if args.video_path:
                break
            else:
                current_video_path = None

    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
