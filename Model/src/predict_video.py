"""
Tahap 5: Prediksi Huruf dari Video
===================================
Menjalankan model BiLSTM terlatih pada satu video dataset dan menampilkan
atau menyimpan video dengan overlay kelas huruf dan confidence.

Jalankan dari folder Model/:
    python src/predict_video.py --video_path dataset/letters/raw/B/B_0001.avi
"""

import argparse
import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from extract_landmarks import extract_landmarks_from_video
from model import BiLSTMModel
from preprocess import normalize_data, pad_or_truncate
from utils import get_device, load_checkpoint


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_input_path(path: str) -> str:
    """Resolve an input path from the current directory or the Model root."""
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(MODEL_ROOT, path)


def load_label_encoder(path: str) -> Tuple[int, list]:
    """Load the persisted class mapping in index order."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label encoder tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as file:
        encoder = json.load(file)

    num_classes = int(encoder["num_classes"])
    idx_to_label = encoder["idx_to_label"]
    class_names = [idx_to_label[str(index)] for index in range(num_classes)]
    return num_classes, class_names


def predict_sequence(
    landmarks: np.ndarray,
    config: dict,
    model: BiLSTMModel,
    device: torch.device,
) -> Tuple[str, float]:
    """Normalize, resize, and classify one extracted landmark sequence."""
    if landmarks.size == 0:
        raise ValueError("Tidak ada landmark yang berhasil diekstrak dari video")

    preprocessing = config["preprocessing"]
    normalized = normalize_data([landmarks], method=preprocessing["normalization"])
    processed = pad_or_truncate(normalized, preprocessing["max_seq_length"])

    expected_features = config["model"]["input_size"]
    actual_features = processed.shape[-1]
    if actual_features != expected_features:
        raise ValueError(
            f"Jumlah fitur tidak cocok: hasil preprocessing={actual_features}, "
            f"model input_size={expected_features}. Pastikan max_num_hands dan "
            "config sama dengan saat training."
        )

    inputs = torch.from_numpy(processed).to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
        confidence, prediction = torch.max(probabilities, dim=0)

    return str(int(prediction.item())), float(confidence.item())


def find_prediction_frame(
    landmarks: np.ndarray,
    motion_threshold: float,
    pause_frames: int,
) -> int:
    """Find the frame where active hand motion is followed by a pause."""
    if motion_threshold <= 0:
        raise ValueError("--motion_threshold harus lebih besar dari 0")
    if pause_frames <= 0:
        raise ValueError("--pause_frames harus lebih besar dari 0")

    motion_detected = False
    quiet_frames = 0
    for frame_index in range(1, len(landmarks)):
        previous = landmarks[frame_index - 1]
        current = landmarks[frame_index]
        valid = np.any(previous != 0, axis=1) & np.any(current != 0, axis=1)
        if not np.any(valid):
            if motion_detected:
                quiet_frames += 1
                if quiet_frames >= pause_frames:
                    return frame_index
            continue

        displacement = np.linalg.norm(current[valid, :2] - previous[valid, :2], axis=1)
        is_moving = float(np.mean(displacement)) >= motion_threshold
        if is_moving:
            motion_detected = True
            quiet_frames = 0
        elif motion_detected:
            quiet_frames += 1
            if quiet_frames >= pause_frames:
                return frame_index

    # Fallback: jika gerakan/jeda tidak cukup jelas, tampilkan hanya pada frame akhir.
    return max(0, len(landmarks) - 1)


def draw_landmarks(frame: np.ndarray, frame_landmarks: np.ndarray) -> None:
    """Draw MediaPipe hand points and connections on one video frame."""
    import mediapipe as mp

    height, width = frame.shape[:2]
    num_hands = frame_landmarks.shape[0] // 21
    colors = [(0, 255, 255), (255, 120, 0)]

    for hand_index in range(num_hands):
        hand = frame_landmarks[hand_index * 21:(hand_index + 1) * 21]
        if np.allclose(hand, 0):
            continue

        points = [
            (int(np.clip(landmark[0], 0, 1) * width), int(np.clip(landmark[1], 0, 1) * height))
            for landmark in hand
        ]
        color = colors[hand_index % len(colors)]
        for start, end in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 7, (255, 255, 255), 1, cv2.LINE_AA)


def default_output_path(video_path: str) -> str:
    """Build the default output path for an input video."""
    filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_prediction.mp4"
    return os.path.join(MODEL_ROOT, "outputs", "letters", "predictions", filename)


def resolve_save_path(user_path: str, video_path: str) -> str:
    """Resolve a blank, directory, or explicit file path from the save prompt."""
    if not user_path.strip():
        return default_output_path(video_path)

    path = resolve_input_path(user_path.strip().strip('"'))
    extension = os.path.splitext(path)[1].lower()
    if os.path.isdir(path) or not extension:
        filename = os.path.basename(default_output_path(video_path))
        return os.path.join(path, filename)
    return path


def annotate_video(
    video_path: str,
    output_path: Optional[str],
    label: str,
    confidence: float,
    landmarks: np.ndarray,
    display: bool,
    display_width: int,
    prediction_frame: int,
) -> None:
    """Replay source video with prediction overlay and optional video output."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka untuk ditampilkan: {video_path}")

    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0:
            fps = 24.0
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError("Ukuran frame video tidak valid")

        if display_width <= 0:
            raise ValueError("--display_width harus lebih besar dari 0")
        scale = min(1.0, display_width / source_width)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        if display:
            cv2.namedWindow("Prediksi Huruf - tekan Q atau Esc untuk keluar", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Prediksi Huruf - tekan Q atau Esc untuk keluar", width, height)

        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter tidak dapat dibuat: {output_path}")

        frame_index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                body_connections = {
                    connection for connection in mp.solutions.pose.POSE_CONNECTIONS
                    if connection[0] >= 11 and connection[1] >= 11
                }
                for start, end in body_connections:
                    start_landmark = pose_results.pose_landmarks.landmark[start]
                    end_landmark = pose_results.pose_landmarks.landmark[end]
                    start_point = (
                        int(np.clip(start_landmark.x, 0, 1) * source_width),
                        int(np.clip(start_landmark.y, 0, 1) * source_height),
                    )
                    end_point = (
                        int(np.clip(end_landmark.x, 0, 1) * source_width),
                        int(np.clip(end_landmark.y, 0, 1) * source_height),
                    )
                    cv2.line(frame, start_point, end_point, (80, 255, 80), 2, cv2.LINE_AA)

                body_indices = {index for connection in body_connections for index in connection}
                for index in body_indices:
                    landmark = pose_results.pose_landmarks.landmark[index]
                    point = (
                        int(np.clip(landmark.x, 0, 1) * source_width),
                        int(np.clip(landmark.y, 0, 1) * source_height),
                    )
                    cv2.circle(frame, point, 4, (255, 80, 80), -1, cv2.LINE_AA)

            if frame_index < len(landmarks):
                draw_landmarks(frame, landmarks[frame_index])
            frame_index += 1

            if scale < 1.0:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 105), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            show_prediction = frame_index - 1 >= prediction_frame
            if show_prediction:
                primary_text = f"Prediksi huruf: {label}"
                secondary_text = f"Confidence: {confidence * 100:.2f}%"
                primary_color = (0, 255, 0)
            else:
                primary_text = "Gerakan sedang berlangsung..."
                secondary_text = "Prediksi tampil setelah tangan berhenti"
                primary_color = (0, 215, 255)

            cv2.putText(
                frame,
                primary_text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                primary_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                secondary_text,
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("Prediksi Huruf - tekan Q atau Esc untuk keluar", frame)
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        pose.close()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi kelas huruf dari video dataset")
    parser.add_argument(
        "--video_path",
        default="dataset/letters/raw/K/K_0001.avi",
        help="Path video input (default: dataset/letters/raw/B/B_0001.avi)",
    )
    parser.add_argument(
        "--model_path",
        default="outputs/letters/models/best_model.pth",
        help="Path checkpoint model (default: outputs/letters/models/best_model.pth)",
    )
    parser.add_argument(
        "--config",
        default="configs/letters.yaml",
        help="Path konfigurasi (default: configs/letters.yaml)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path video hasil anotasi (opsional, codec mp4v)",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=960,
        help="Lebar maksimum window/output dalam piksel (default: 960)",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.003,
        help="Ambang perpindahan tangan per frame (default: 0.003)",
    )
    parser.add_argument(
        "--pause_frames",
        type=int,
        default=8,
        help="Jumlah frame diam sebelum prediksi ditampilkan (default: 8)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Jangan membuka window; gunakan bersama --output_path untuk mode headless",
    )
    args = parser.parse_args()
    args.video_path = resolve_input_path(args.video_path)
    args.model_path = resolve_input_path(args.model_path)
    args.config = resolve_input_path(args.config)
    if args.output_path:
        args.output_path = resolve_input_path(args.output_path)

    if not args.video_path.lower().endswith(VIDEO_EXTENSIONS):
        print(f"[WARN] Ekstensi video tidak umum: {args.video_path}")
    if not os.path.isfile(args.video_path):
        print(f"[ERROR] Video tidak ditemukan: {args.video_path}")
        return 1
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model tidak ditemukan: {args.model_path}")
        return 1
    if not os.path.isfile(args.config):
        print(f"[ERROR] Config tidak ditemukan: {args.config}")
        return 1

    hands = None
    try:
        with open(args.config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        encoder_path = resolve_input_path(
            os.path.join(config["paths"]["processed"], "label_encoder.json")
        )
        num_classes, class_names = load_label_encoder(encoder_path)
        device = get_device()
        model = BiLSTMModel(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=num_classes,
            dropout=config["model"]["dropout"],
        ).to(device)
        load_checkpoint(args.model_path, model, device=device)

        import mediapipe as mp

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config["landmark"]["max_num_hands"],
            min_detection_confidence=config["landmark"].get("min_detection_confidence", 0.5),
            min_tracking_confidence=config["landmark"].get("min_tracking_confidence", 0.5),
        )
        landmarks = extract_landmarks_from_video(
            args.video_path,
            hands,
            max_num_hands=config["landmark"]["max_num_hands"],
        )
        label_index, confidence = predict_sequence(landmarks, config, model, device)
        label = class_names[int(label_index)]
        prediction_frame = find_prediction_frame(
            landmarks,
            motion_threshold=args.motion_threshold,
            pause_frames=args.pause_frames,
        )
        print(f"[RESULT] Prediksi: {label} (confidence: {confidence * 100:.2f}%)")
        print(f"[INFO] Prediksi ditampilkan mulai frame {prediction_frame + 1}")

        if args.no_display and not args.output_path:
            print("[INFO] Video tidak dibuat karena --no_display digunakan tanpa --output_path")
        else:
            annotate_video(
                args.video_path,
                args.output_path,
                label,
                confidence,
                landmarks,
                display=not args.no_display,
                display_width=args.display_width,
                prediction_frame=prediction_frame,
            )
            if args.output_path:
                print(f"[SAVED] Video hasil prediksi: {args.output_path}")
            elif not args.no_display:
                answer = input("Apakah ingin disave hasil video ini? (y/n): ").strip().lower()
                if answer in ("y", "ya"):
                    default_path = default_output_path(args.video_path)
                    user_path = input(
                        "Direktori/file tujuan "
                        f"(Enter untuk {default_path}): "
                    )
                    save_path = resolve_save_path(user_path, args.video_path)
                    annotate_video(
                        args.video_path,
                        save_path,
                        label,
                        confidence,
                        landmarks,
                        display=False,
                        display_width=args.display_width,
                        prediction_frame=prediction_frame,
                    )
                    print(f"[SAVED] Video hasil prediksi: {save_path}")
                else:
                    print("[INFO] Video tidak disimpan")
    except (KeyError, OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"[ERROR] Prediksi gagal: {error}")
        return 1
    finally:
        if hands is not None:
            hands.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
