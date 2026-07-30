"""
Realtime inference huruf dari webcam.

Jalankan dari folder Model/:
    python -m src.letters.predict_webcam

Prediksi memakai landmark tangan saja. Skeleton tubuh MediaPipe Pose hanya
untuk visualisasi dan tidak diberikan ke model.
"""

import argparse
import os
from collections import Counter, deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from src.common.utils import get_device, load_checkpoint
from src.letters.model import BiLSTMModel
from src.letters.predict_video import (
    MODEL_ROOT,
    draw_landmarks,
    load_label_encoder,
    predict_sequence,
    resolve_input_path,
)

WINDOW_NAME = "Prediksi Huruf Webcam - Q/Esc untuk keluar"


def extract_webcam_landmarks(results, max_num_hands: int) -> np.ndarray:
    """Return fixed [Left, Right] hand slots, matching dataset extraction."""
    slots = {"Left": np.zeros((21, 3), dtype=np.float32), "Right": np.zeros((21, 3), dtype=np.float32)}
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            if label in slots:
                slots[label] = np.asarray(
                    [[point.x, point.y, point.z] for point in hand.landmark],
                    dtype=np.float32,
                )

    if max_num_hands == 1:
        first = next((slot for slot in slots.values() if not np.allclose(slot, 0)), np.zeros((21, 3)))
        return first.astype(np.float32)
    return np.vstack([slots["Left"], slots["Right"]]).astype(np.float32)


def hand_motion(previous: Optional[np.ndarray], current: np.ndarray) -> tuple[bool, bool]:
    """Return (hands_present, is_moving) from two consecutive frames."""
    present = bool(np.any(current != 0))
    if previous is None:
        return present, False
    valid = np.any(previous != 0, axis=1) & np.any(current != 0, axis=1)
    if not np.any(valid):
        return present, False
    displacement = np.linalg.norm(current[valid, :2] - previous[valid, :2], axis=1)
    return present, float(np.mean(displacement))


def draw_pose_body(frame: np.ndarray, pose_results, width: int, height: int) -> None:
    """Draw body-only Pose landmarks, excluding all face connections."""
    if not pose_results.pose_landmarks:
        return
    body_connections = {
        connection for connection in mp.solutions.pose.POSE_CONNECTIONS
        if connection[0] >= 11 and connection[1] >= 11
    }
    landmarks = pose_results.pose_landmarks.landmark
    for start, end in body_connections:
        first, second = landmarks[start], landmarks[end]
        p1 = (int(np.clip(first.x, 0, 1) * width), int(np.clip(first.y, 0, 1) * height))
        p2 = (int(np.clip(second.x, 0, 1) * width), int(np.clip(second.y, 0, 1) * height))
        cv2.line(frame, p1, p2, (80, 255, 80), 2, cv2.LINE_AA)
    body_indices = {index for connection in body_connections for index in connection}
    for index in body_indices:
        point = landmarks[index]
        center = (int(np.clip(point.x, 0, 1) * width), int(np.clip(point.y, 0, 1) * height))
        cv2.circle(frame, center, 4, (255, 80, 80), -1, cv2.LINE_AA)


def put_status(frame: np.ndarray, title: str, detail: str, color: tuple[int, int, int]) -> None:
    """Draw status panel and controls."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q/Esc: keluar | Gerakkan tangan untuk huruf berikutnya", (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def prediction_consensus(
    predictions: list[tuple[int, float]],
    vote_window: int,
    vote_ratio: float,
    stable_confidence: float,
) -> tuple[Optional[int], Optional[int], float, float]:
    """Return (stable, candidate, vote ratio, mean confidence) from recent predictions."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi huruf secara realtime dari webcam")
    parser.add_argument("--camera_index", type=int, default=0, help="Index webcam OpenCV (default: 0)")
    parser.add_argument("--config", default="configs/letters.yaml", help="Path config model")
    parser.add_argument("--model_path", default="outputs/letters/models/best_model.pth", help="Path checkpoint")
    parser.add_argument("--display_width", type=int, default=960, help="Lebar maksimum window (default: 960)")
    parser.add_argument("--motion_threshold", type=float, default=0.003, help="Ambang gerakan tangan (default: 0.003)")
    parser.add_argument("--pause_frames", type=int, default=8, help="Frame diam untuk fallback prediksi (default: 8)")
    parser.add_argument("--result_frames", type=int, default=45, help="Durasi hasil sebelum siap membaca huruf berikutnya (default: 45)")
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Confidence minimum untuk menampilkan hasil")
    parser.add_argument("--min_recording_frames", type=int, default=30, help="Frame minimum sebelum voting dimulai (default: 30)")
    parser.add_argument("--prediction_interval", type=int, default=3, help="Interval frame antar prediksi realtime (default: 3)")
    parser.add_argument("--vote_window", type=int, default=5, help="Jumlah prediksi terbaru untuk voting (default: 5)")
    parser.add_argument("--vote_ratio", type=float, default=0.8, help="Rasio vote minimum untuk mengunci hasil (default: 0.8)")
    parser.add_argument("--stable_confidence", type=float, default=0.8, help="Rata-rata confidence minimum untuk mengunci hasil (default: 0.8)")
    parser.add_argument("--rearm_motion_frames", type=int, default=3, help="Frame gerak untuk membaca huruf berikutnya (default: 3)")
    parser.add_argument("--mirror", action="store_true", help="Balik tampilan seperti cermin; tidak disarankan untuk input model")
    args = parser.parse_args()

    positive_values = {
        "display_width": args.display_width,
        "motion_threshold": args.motion_threshold,
        "pause_frames": args.pause_frames,
        "result_frames": args.result_frames,
        "min_recording_frames": args.min_recording_frames,
        "prediction_interval": args.prediction_interval,
        "vote_window": args.vote_window,
        "rearm_motion_frames": args.rearm_motion_frames,
    }
    invalid_positive = [name for name, value in positive_values.items() if value <= 0]
    if invalid_positive:
        print(f"[ERROR] {', '.join(invalid_positive)} harus lebih besar dari 0")
        return 1
    probability_values = {
        "min_confidence": args.min_confidence,
        "vote_ratio": args.vote_ratio,
        "stable_confidence": args.stable_confidence,
    }
    invalid_probability = [
        name for name, value in probability_values.items() if not 0 <= value <= 1
    ]
    if invalid_probability:
        print(f"[ERROR] {', '.join(invalid_probability)} harus berada di antara 0 dan 1")
        return 1

    config_path = resolve_input_path(args.config)
    model_path = resolve_input_path(args.model_path)
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config tidak ditemukan: {config_path}")
        return 1
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return 1

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        print(f"[ERROR] Webcam dengan index {args.camera_index} tidak dapat dibuka")
        return 1

    hands = pose = None
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        num_classes, class_names = load_label_encoder(
            resolve_input_path(os.path.join(config["paths"]["processed"], "label_encoder.json"))
        )
        device = get_device()
        model = BiLSTMModel(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=num_classes,
            dropout=config["model"]["dropout"],
        ).to(device)
        load_checkpoint(model_path, model, device=device)

        max_num_hands = config["landmark"]["max_num_hands"]
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=config["landmark"].get("min_detection_confidence", 0.5),
            min_tracking_confidence=config["landmark"].get("min_tracking_confidence", 0.5),
        )
        pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
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
        print("[INFO] Webcam aktif. Hasil akan terkunci saat voting prediksi stabil.")
        print("[INFO] Setelah hasil tampil, cukup gerakkan tangan untuk membaca huruf berikutnya.")

        while True:
            ok, frame = capture.read()
            if not ok:
                print("[ERROR] Frame webcam tidak dapat dibaca")
                return 1

            source_height, source_width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)
            current = extract_webcam_landmarks(hand_results, max_num_hands)
            present, motion = hand_motion(previous, current)
            previous = current.copy()

            draw_pose_body(frame, pose_results, source_width, source_height)
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
                    candidate_ratio = 0.0
                title, detail, color = "Siap mulai gerakan", "Gerakkan tangan untuk membaca huruf", (0, 215, 255)
            elif state == "RECORDING":
                buffer.append(current.copy())
                frames_since_prediction += 1
                if len(buffer) > config["preprocessing"]["max_seq_length"] * 2:
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
                        result_label = class_names[stable_label]
                        result_confidence = candidate_confidence
                        result_countdown = args.result_frames
                        state = "RESULT"
                        print(
                            f"[RESULT] Voting stabil: {result_label} "
                            f"(vote: {candidate_ratio * 100:.0f}%, "
                            f"confidence: {result_confidence * 100:.2f}%)"
                        )

                if (
                    state == "RECORDING"
                    and len(prediction_history) >= args.vote_window
                    and quiet_frames >= args.pause_frames
                ):
                    label_index, result_confidence = predict_sequence(
                        np.asarray(buffer), config, model, device
                    )
                    result_label = class_names[int(label_index)]
                    result_countdown = args.result_frames
                    state = "RESULT"
                    print(
                        f"[RESULT] Fallback jeda: {result_label} "
                        f"(confidence: {result_confidence * 100:.2f}%)"
                    )

                if state == "RESULT":
                    title = f"Prediksi huruf: {result_label}"
                    detail = f"Confidence: {result_confidence * 100:.2f}% | Hasil terkunci"
                    color = (0, 255, 0) if result_confidence >= args.min_confidence else (0, 165, 255)
                elif len(buffer) < args.min_recording_frames:
                    title = "Gerakan sedang direkam"
                    detail = f"Mengumpulkan frame: {len(buffer)}/{args.min_recording_frames}"
                    color = (0, 215, 255)
                elif candidate_label:
                    title = f"Kandidat realtime: {candidate_label}"
                    detail = (
                        f"Vote: {candidate_ratio * 100:.0f}% | "
                        f"Confidence: {candidate_confidence * 100:.2f}%"
                    )
                    color = (0, 215, 255)
                else:
                    title, detail, color = "Menganalisis gerakan", "Menunggu voting prediksi", (0, 215, 255)
            elif state == "RESULT":
                title = f"Prediksi huruf: {result_label}"
                detail = f"Confidence: {result_confidence * 100:.2f}% | Hasil terkunci"
                color = (0, 255, 0) if result_confidence >= args.min_confidence else (0, 165, 255)
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
                    candidate_ratio = 0.0
            else:
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
                    candidate_ratio = 0.0
                    title, detail, color = "Gerakan baru terdeteksi", "Merekam huruf berikutnya", (0, 215, 255)
                else:
                    title = "Siap untuk huruf berikutnya"
                    detail = "Gerakkan tangan ke pose baru; tidak perlu diturunkan"
                    color = (0, 215, 255)

            if state != "RESULT" and result_label:
                # Hasil lama tidak boleh terlihat sebelum siklus baru selesai.
                result_label = ""
                result_confidence = 0.0
            if args.mirror:
                frame = cv2.flip(frame, 1)
            scale = min(1.0, args.display_width / source_width)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(source_width * scale), int(source_height * scale)), interpolation=cv2.INTER_AREA)
            put_status(frame, title, detail, color)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except (KeyError, OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"[ERROR] Webcam inference gagal: {error}")
        return 1
    finally:
        capture.release()
        if hands is not None:
            hands.close()
        if pose is not None:
            pose.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
