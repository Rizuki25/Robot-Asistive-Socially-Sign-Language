"""
Inference video tunggal menggunakan model gabungan 36 kelas.

Jalankan dari folder Model/:
    python -m src.combined.predict_video --video_path <path_video.mp4> --config configs/combined_90.yaml
"""

import argparse
import json
import os
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from src.common.utils import get_device
from src.combined.model import CombinedBiLSTM
from src.combined.preprocess import extract_combined_features, pad_features

WINDOW_NAME = "Prediksi Video (36 Kelas Gabungan) - Q/Esc untuk keluar"
MODEL_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def resolve_input_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(MODEL_ROOT, path)


def load_label_encoder(path: str) -> Tuple[int, list]:
    resolved = resolve_input_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        encoder = json.load(f)
    num_classes = int(encoder["num_classes"])
    idx_to_label = encoder["idx_to_label"]
    class_names = [idx_to_label[str(idx)] for idx in range(num_classes)]
    return num_classes, class_names


def extract_landmarks_from_video(
    video_path: str,
    max_num_hands: int = 2,
) -> Tuple[np.ndarray, list, float, int, int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Video tidak dapat dibuka: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
    features = extract_combined_features(landmarks)
    data, lengths = pad_features([features], config["preprocessing"]["max_seq_length"])
    inputs = torch.from_numpy(data).to(device)
    lengths_t = torch.from_numpy(lengths).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(inputs, lengths_t), dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return str(int(pred.item())), float(conf.item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict single video with 36-Class Combined Model")
    parser.add_argument("--video_path", required=True, help="Path ke file video input")
    parser.add_argument("--config", default="configs/combined_90.yaml", help="Path config gabungan")
    parser.add_argument("--model_path", default=None, help="Path checkpoint model (default: best_model.pth)")
    parser.add_argument("--display_width", type=int, default=640, help="Lebar display window")
    parser.add_argument("--no_display", action="store_true", help="Jalankan headless tanpa window")
    args = parser.parse_args()

    video_path = resolve_input_path(args.video_path)
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video tidak ditemukan: {video_path}")
        return 1

    with open(resolve_input_path(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()
    paths = config["paths"]
    encoder_path = resolve_input_path(os.path.join(paths["processed"], "label_encoder.json"))
    num_classes, class_names = load_label_encoder(encoder_path)

    model_path = args.model_path or os.path.join(paths["models"], "best_model.pth")
    model_path = resolve_input_path(model_path)

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

    print(f"[INFO] Memproses video: {video_path}")
    landmarks, frames, fps, width, height = extract_landmarks_from_video(video_path)
    label_idx_str, confidence = predict_sequence(landmarks, config, model, device)
    predicted_label = class_names[int(label_idx_str)]

    print("\n=======================================================")
    print(f"HASIL PREDIKSI VIDEO (36 KELAS):")
    print(f"  Prediksi Label : {predicted_label}")
    print(f"  Confidence     : {confidence * 100:.2f}%")
    print("=======================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

