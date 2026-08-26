"""
Realtime inference webcam menggunakan model gabungan 36 kelas (Huruf & Kata).

Jalankan dari folder Model/:
    python -m src.combined.predict_webcam --config configs/combined_90.yaml
"""

import argparse
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
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
from src.combined.predict_video import (
    MODEL_ROOT,
    load_label_encoder,
    predict_sequence,
    resolve_input_path,
)

WINDOW_NAME = "Prediksi 36 Kelas (Huruf & Kata) - Q/Esc untuk keluar"


def extract_webcam_landmarks(results, max_num_hands: int = 2) -> np.ndarray:
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
    if max_num_hands == 1:
        first = next(
            (slot for slot in slots.values() if not np.allclose(slot, 0)),
            np.zeros((21, 3), dtype=np.float32),
        )
        return first.astype(np.float32)
    return np.vstack([slots["Left"], slots["Right"]]).astype(np.float32)


def hand_motion(previous: Optional[np.ndarray], current: np.ndarray) -> tuple[bool, float]:
    present = bool(np.any(current != 0))
    if previous is None:
        return present, 0.0
    valid = np.any(previous != 0, axis=1) & np.any(current != 0, axis=1)
    if not np.any(valid):
        return present, 0.0
    displacement = np.linalg.norm(current[valid, :2] - previous[valid, :2], axis=1)
    return present, float(np.mean(displacement))


def draw_landmarks(frame: np.ndarray, frame_landmarks: np.ndarray) -> None:
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


def put_status(frame: np.ndarray, title: str, detail: str, color: tuple[int, int, int], fps: float = 0.0) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.0f}", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Model 36 Kelas (Huruf & Kata)", (75, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "[Q/Esc]: Keluar", (w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, title, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


def prediction_consensus(
    predictions: list[tuple[int, float]],
    vote_window: int,
    vote_ratio: float,
    stable_confidence: float,
) -> tuple[Optional[int], Optional[int], float, float]:
    if not predictions:
        return None, None, 0.0, 0.0

    counts = Counter(label for label, _ in predictions)
    confidence_by_label = {
        label: float(np.mean([confidence for item, confidence in predictions if item == label]))
        for label in counts
    }
    candidate = max(
        counts,
        key=lambda label: (counts[label], confidence_by_label[label], -label),
    )
    candidate_ratio = counts[candidate] / len(predictions)
    candidate_confidence = confidence_by_label[candidate]
    stable = (
        candidate
        if len(predictions) >= vote_window
        and candidate_ratio >= vote_ratio
        and candidate_confidence >= stable_confidence
        else None
    )
    return stable, candidate, candidate_ratio, candidate_confidence


class CombinedAudioPlayer:
    def __init__(self, letters_dir: str, words_dir: str):
        self.letters_dir = resolve_input_path(letters_dir)
        self.words_dir = resolve_input_path(words_dir)
        self._missing_warnings = set()
        self.available = winsound is not None

    def _find_audio(self, label: str) -> Optional[str]:
        candidates = [
            os.path.join(self.words_dir, f"{label}.wav"),
            os.path.join(self.words_dir, f"{label.lower()}.wav"),
            os.path.join(self.letters_dir, f"{label.upper()}.wav"),
            os.path.join(self.letters_dir, f"{label}.wav"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def play(self, label: str) -> None:
        if not self.available:
            return
        path = self._find_audio(label)
        if path is None:
            if label not in self._missing_warnings:
                print(f"[INFO] Audio '{label}' belum tersedia.")
                self._missing_warnings.add(label)
            return
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except RuntimeError as error:
            print(f"[WARN] Gagal memutar audio ({path}): {error}")

    def close(self) -> None:
        if self.available:
            winsound.PlaySound(None, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi 36 kelas realtime dari webcam")
    parser.add_argument("--camera_index", type=int, default=0, help="Index webcam (default: 0)")
    parser.add_argument("--config", default="configs/combined_90.yaml", help="Path config gabungan")
    parser.add_argument("--model_path", default=None, help="Path checkpoint (default: best_model.pth)")
    parser.add_argument("--display_width", type=int, default=640, help="Lebar window (default: 640)")
    parser.add_argument("--display_height", type=int, default=480, help="Tinggi window (default: 480)")
    parser.add_argument("--motion_threshold", type=float, default=0.001, help="Ambang gerakan (default: 0.001)")
    parser.add_argument("--pause_frames", type=int, default=15, help="Frame jeda (default: 15)")
    parser.add_argument("--result_frames", type=int, default=45, help="Durasi hasil terkunci tampil (default: 45)")
    parser.add_argument("--min_recording_frames", type=int, default=35, help="Frame minimum sebelum voting (default: 35)")
    parser.add_argument("--prediction_interval", type=int, default=3, help="Interval inferensi berkala (default: 3)")
    parser.add_argument("--vote_window", type=int, default=5, help="Ukuran window voting (default: 5)")
    parser.add_argument("--vote_ratio", type=float, default=0.8, help="Rasio vote konsensus (default: 0.8)")
    parser.add_argument("--stable_confidence", type=float, default=0.8, help="Confidence minimum kunci (default: 0.8)")
    parser.add_argument("--rearm_motion_frames", type=int, default=3, help="Frame gerak pemicu (default: 3)")
    parser.add_argument("--letters_audio_dir", default="assets/letters_audio", help="Folder audio huruf")
    parser.add_argument("--words_audio_dir", default="assets/words_audio", help="Folder audio kata")
    parser.add_argument("--no_speech", action="store_true", help="Nonaktifkan suara audio")
    parser.add_argument("--mirror", action="store_true", help="Cerminkan tampilan kamera")
    args = parser.parse_args()

    config_path = resolve_input_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = args.model_path or os.path.join(config["paths"]["models"], "best_model.pth")
    model_path = resolve_input_path(model_path)
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model checkpoint tidak ditemukan: {model_path}")
        return 1

    encoder_path = resolve_input_path(os.path.join(config["paths"]["processed"], "label_encoder.json"))
    num_classes, class_names = load_label_encoder(encoder_path)
    device = get_device()

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
        audio_player = CombinedAudioPlayer(args.letters_audio_dir, args.words_audio_dir)

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        print(f"[ERROR] Webcam index {args.camera_index} tidak dapat dibuka")
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)

    display_w = args.display_width
    display_h = args.display_height

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_w, display_h)

    previous = None
    buffer = []
    state = "WAITING"
    quiet_frames = 0
    rearm_motion_count = 0
    result_countdown = 0
    result_label = ""
    result_confidence = 0.0
    candidate_label = ""
    candidate_confidence = 0.0
    candidate_ratio = 0.0
    prediction_history = deque(maxlen=args.vote_window)
    frames_since_prediction = 0
    rearm_motion_buffer = []

    fps_val = 0.0
    frame_counter = 0
    fps_start_time = time.time()

    def lock_result(label: str, confidence: float, source: str) -> None:
        nonlocal result_label, result_confidence, result_countdown, state
        result_label = label
        result_confidence = confidence
        result_countdown = args.result_frames
        state = "RESULT"
        print(f"[RESULT] {source}: {result_label} (confidence: {result_confidence * 100:.2f}%)")
        if audio_player is not None:
            audio_player.play(result_label)

    print("\n[INFO] Webcam 36-Class aktif. Hasil akan terkunci saat voting stabil.\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_counter += 1
            if frame_counter >= 15:
                now = time.time()
                fps_val = frame_counter / (now - fps_start_time)
                fps_start_time = now
                frame_counter = 0

            if (frame.shape[1], frame.shape[0]) != (display_w, display_h):
                frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = mp_hands.process(rgb)
            current = extract_webcam_landmarks(hand_results, max_num_hands=2)
            present, motion = hand_motion(previous, current)
            previous = current.copy()

            draw_landmarks(frame, current)

            if state == "WAITING":
                if present and motion >= args.motion_threshold:
                    state = "RECORDING"
                    buffer = [current.copy()]
                    quiet_frames = 0
                    frames_since_prediction = 0
                    prediction_history.clear()
                    candidate_label = ""
                    candidate_confidence = 0.0
                title = "Siap Mulai Gerakan"
                detail = "Peragakan Huruf atau Kata"
                color = (0, 215, 255)

            elif state == "RECORDING":
                buffer.append(current.copy())
                frames_since_prediction += 1
                max_buffer_size = 180
                if len(buffer) > max_buffer_size:
                    buffer.pop(0)

                if present and motion < args.motion_threshold:
                    quiet_frames += 1
                elif motion >= args.motion_threshold:
                    quiet_frames = 0

                if (
                    len(buffer) >= args.min_recording_frames
                    and frames_since_prediction >= args.prediction_interval
                ):
                    label_index, confidence = predict_sequence(
                        np.asarray(buffer), config, model, device
                    )
                    frames_since_prediction = 0
                    prediction_history.append((int(label_index), confidence))
                    stable_label, candidate, candidate_ratio, candidate_confidence = (
                        prediction_consensus(
                            list(prediction_history),
                            args.vote_window,
                            args.vote_ratio,
                            args.stable_confidence,
                        )
                    )
                    candidate_label = class_names[candidate] if candidate is not None else ""
                    if stable_label is not None:
                        lock_result(
                            class_names[stable_label],
                            candidate_confidence,
                            f"Voting stabil ({candidate_ratio * 100:.0f}%)",
                        )

                if (
                    state == "RECORDING"
                    and len(buffer) >= args.min_recording_frames
                    and quiet_frames >= args.pause_frames
                ):
                    label_index, fallback_confidence = predict_sequence(
                        np.asarray(buffer), config, model, device
                    )
                    lock_result(
                        class_names[int(label_index)],
                        fallback_confidence,
                        "Fallback jeda",
                    )

                if state == "RESULT":
                    title = f"Prediksi: {result_label}"
                    detail = f"Confidence: {result_confidence * 100:.1f}% | Terkunci"
                    color = (0, 255, 0)
                elif len(buffer) < args.min_recording_frames:
                    title = "Merekam Gerakan..."
                    detail = f"Mengumpulkan frame: {len(buffer)}/{args.min_recording_frames}"
                    color = (0, 215, 255)
                elif candidate_label:
                    title = f"Kandidat: {candidate_label}"
                    detail = f"Vote: {candidate_ratio * 100:.0f}% | Conf: {candidate_confidence * 100:.1f}%"
                    color = (0, 215, 255)
                else:
                    title = "Menganalisis Gerakan..."
                    detail = "Menunggu konsensus prediksi"
                    color = (0, 215, 255)

            elif state == "RESULT":
                title = f"Prediksi: {result_label}"
                detail = f"Confidence: {result_confidence * 100:.1f}% | Terkunci"
                color = (0, 255, 0)
                result_countdown -= 1
                if result_countdown <= 0:
                    state = "REARM"
                    buffer = []
                    quiet_frames = 0
                    rearm_motion_count = 0
                    rearm_motion_buffer = []
                    prediction_history.clear()
                    candidate_label = ""
                    candidate_confidence = 0.0

            else:  # REARM
                if present and motion >= args.motion_threshold:
                    rearm_motion_count += 1
                    rearm_motion_buffer.append(current.copy())
                else:
                    rearm_motion_count = 0
                    rearm_motion_buffer = []

                if rearm_motion_count >= args.rearm_motion_frames:
                    state = "RECORDING"
                    buffer = list(rearm_motion_buffer)
                    quiet_frames = 0
                    frames_since_prediction = 0
                    prediction_history.clear()
                    candidate_label = ""
                    candidate_confidence = 0.0
                    title = "Gerakan Baru Terdeteksi"
                    detail = "Merekam isyarat berikutnya..."
                    color = (0, 215, 255)
                else:
                    title = "Siap Isyarat Berikutnya"
                    detail = "Gerakkan tangan ke isyarat baru"
                    color = (0, 215, 255)

            if state != "RESULT" and result_label:
                result_label = ""
                result_confidence = 0.0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            put_status(frame, title, detail, color, fps=fps_val)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        capture.release()
        if audio_player is not None:
            audio_player.close()
        mp_hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

