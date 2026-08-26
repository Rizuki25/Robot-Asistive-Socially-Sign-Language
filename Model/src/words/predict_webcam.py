"""
Realtime inference kata (words) dari webcam menggunakan BiLSTM motion-aware.

Jalankan dari folder Model/:
    python -m src.words.predict_webcam --config configs/words_motion_90.yaml

Prediksi memakai sequence landmark tangan (140 fitur motion-aware).
Skeleton tubuh MediaPipe Pose digambar sebagai visualisasi dan tidak diberikan ke model.
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
from typing import Optional

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
from src.words.model import WordMotionBiLSTM
from src.words.predict_video import (
    MODEL_ROOT,
    draw_landmarks,
    load_label_encoder,
    predict_sequence,
    resolve_input_path,
)

WINDOW_NAME = "Prediksi Kata Webcam - Q/Esc untuk keluar"


def extract_webcam_landmarks(results, max_num_hands: int = 2) -> np.ndarray:
    """Return fixed [Left, Right] hand slots, matching dataset extraction."""
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
    """Return (hands_present, displacement) from two consecutive frames."""
    present = bool(np.any(current != 0))
    if previous is None:
        return present, 0.0
    valid = np.any(previous != 0, axis=1) & np.any(current != 0, axis=1)
    if not np.any(valid):
        return present, 0.0
    displacement = np.linalg.norm(current[valid, :2] - previous[valid, :2], axis=1)
    return present, float(np.mean(displacement))


def draw_pose_body(frame: np.ndarray, pose_results, width: int, height: int) -> None:
    """Draw body-only Pose landmarks, excluding all face connections."""
    if not pose_results.pose_landmarks:
        return
    body_connections = {
        connection
        for connection in mp.solutions.pose.POSE_CONNECTIONS
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
    """Draw status panel and controls overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Q/Esc: keluar | Gerakkan tangan untuk kata berikutnya",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


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


class RecordedAudioPlayer:
    """Play one user-recorded WAV file per locked prediction on Windows."""

    def __init__(self, audio_dir: str, class_names: list[str]):
        self.audio_dir = resolve_input_path(audio_dir)
        self._missing_warnings = set()
        self.available = winsound is not None

        if not self.available:
            print("[WARN] Pemutaran WAV hanya tersedia di Windows; prediksi tetap berjalan")
            return

        missing = [
            label for label in class_names
            if not self._find_audio_file(label)
        ]
        if missing:
            print(
                f"[INFO] File audio kata di {self.audio_dir}: "
                f"{len(class_names) - len(missing)}/{len(class_names)} tersedia "
                f"(belum ada: {', '.join(missing)})"
            )
        else:
            print(f"[INFO] Rekaman suara kata siap: {self.audio_dir}")

    def _find_audio_file(self, label: str) -> Optional[str]:
        candidates = [
            os.path.join(self.audio_dir, f"{label}.wav"),
            os.path.join(self.audio_dir, f"{label.lower()}.wav"),
            os.path.join(self.audio_dir, f"{label.upper()}.wav"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def play(self, label: str) -> None:
        """Play a label asynchronously; missing files warn only once."""
        if not self.available:
            return
        path = self._find_audio_file(label)
        if path is None:
            if label not in self._missing_warnings:
                print(f"[INFO] Audio suara kata '{label}' belum tersedia di {self.audio_dir}")
                self._missing_warnings.add(label)
            return
        try:
            winsound.PlaySound(
                path,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
        except RuntimeError as error:
            if path not in self._missing_warnings:
                print(f"[WARN] Rekaman suara gagal diputar ({path}): {error}")
                self._missing_warnings.add(path)

    def close(self) -> None:
        """Stop an asynchronous WAV that is still playing."""
        if self.available:
            winsound.PlaySound(None, 0)


class MobileAppPublisher:
    """Publish locked predictions without blocking webcam inference."""

    def __init__(
        self,
        server_url: str,
        room_id: str,
        api_key: str,
    ):
        self.server_url = server_url.rstrip("/")
        self.room_id = room_id
        self.api_key = api_key
        self._last_warning_time = 0.0
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._result_thread = threading.Thread(
            target=self._result_worker,
            name="mobile-word-result-publisher",
            daemon=True,
        )
        self._result_thread.start()
        print(
            f"[INFO] Sinkronisasi app aktif: {self.server_url} "
            f"(room: {self.room_id}, hasil prediksi kata)"
        )

    def _headers(self, content_type: str) -> dict[str, str]:
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["x-model-api-key"] = self.api_key
        return headers

    def _post(self, path: str, data: bytes, content_type: str) -> None:
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            headers=self._headers(content_type),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            response.read()

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_time >= 5:
            print(f"[WARN] Sinkronisasi app gagal: {message}")
            self._last_warning_time = now

    def publish_prediction(self, label: str, confidence: float, source: str) -> None:
        self._result_queue.put(
            {
                "roomId": self.room_id,
                "text": label,
                "confidence": float(confidence),
                "source": f"predict_webcam_words:{source}",
            }
        )

    def _result_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = self._result_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._post(
                    "/api/sign-result",
                    json.dumps(result).encode("utf-8"),
                    "application/json",
                )
                print(
                    f"[SYNC] Hasil dikirim ke app: {result['text']} "
                    f"({result['confidence'] * 100:.2f}%)"
                )
            except (OSError, urllib.error.URLError) as error:
                self._warn(str(error))

    def close(self) -> None:
        self._stop_event.set()
        self._result_thread.join(timeout=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi kata secara realtime dari webcam")
    parser.add_argument("--camera_index", type=int, default=0, help="Index webcam OpenCV (default: 0)")
    parser.add_argument("--config", default="configs/words_motion_90.yaml", help="Path config model")
    parser.add_argument("--model_path", default=None, help="Path checkpoint (default: otomatis dari config)")
    parser.add_argument("--display_width", type=int, default=960, help="Lebar maksimum window (default: 960)")
    parser.add_argument("--motion_threshold", type=float, default=0.001, help="Ambang gerakan tangan (default: 0.001)")
    parser.add_argument("--pause_frames", type=int, default=15, help="Frame diam untuk fallback prediksi (default: 15)")
    parser.add_argument("--result_frames", type=int, default=45, help="Durasi hasil sebelum siap membaca kata berikutnya (default: 45)")
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Confidence minimum untuk menampilkan hasil")
    parser.add_argument("--min_recording_frames", type=int, default=45, help="Frame minimum sebelum voting dimulai (default: 45)")
    parser.add_argument("--prediction_interval", type=int, default=3, help="Interval frame antar prediksi realtime (default: 3)")
    parser.add_argument("--vote_window", type=int, default=5, help="Jumlah prediksi terbaru untuk voting (default: 5)")
    parser.add_argument("--vote_ratio", type=float, default=0.8, help="Rasio vote minimum untuk mengunci hasil (default: 0.8)")
    parser.add_argument("--stable_confidence", type=float, default=0.8, help="Rata-rata confidence minimum untuk mengunci hasil (default: 0.8)")
    parser.add_argument("--rearm_motion_frames", type=int, default=3, help="Frame gerak untuk membaca kata berikutnya (default: 3)")
    parser.add_argument("--audio_dir", default="assets/words_audio", help="Folder rekaman audio kata (default: assets/words_audio)")
    parser.add_argument("--no_speech", action="store_true", help="Nonaktifkan pemutaran rekaman suara")
    parser.add_argument("--mirror", action="store_true", help="Balik tampilan seperti cermin; tidak disarankan untuk input model")
    parser.add_argument(
        "--server_url",
        default=os.environ.get("MODEL_SERVER_URL", ""),
        help="URL backend app, misalnya http://localhost:3001 (default: MODEL_SERVER_URL)",
    )
    parser.add_argument("--room_id", default="demo-ta", help="Room tujuan app mobile (default: demo-ta)")
    parser.add_argument(
        "--model_api_key",
        default=os.environ.get("MODEL_API_KEY", ""),
        help="API key opsional yang sama dengan MODEL_API_KEY di backend",
    )
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
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config tidak ditemukan: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_path = args.model_path or os.path.join(
        config["paths"]["models"], "best_model.pth"
    )
    model_path = resolve_input_path(model_path)
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model checkpoint tidak ditemukan: {model_path}")
        return 1

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        print(f"[ERROR] Webcam dengan index {args.camera_index} tidak dapat dibuka")
        return 1

    # Coba request resolusi 1280x720 dari kamera
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Baca frame uji untuk memastikan ukuran sumber
    ret_test, frame_test = capture.read()
    if not ret_test or frame_test is None:
        print("[ERROR] Tidak dapat membaca frame awal dari webcam")
        return 1

    source_height, source_width = frame_test.shape[:2]
    scale = args.display_width / source_width if source_width > 0 else 1.0
    display_w = max(1, int(round(source_width * scale)))
    display_h = max(1, int(round(source_height * scale)))

    hands = pose = None
    audio_player = None
    app_publisher = None
    try:
        encoder_path = resolve_input_path(
            os.path.join(config["paths"]["processed"], "label_encoder.json")
        )
        num_classes, class_names = load_label_encoder(encoder_path)
        device = get_device()

        model = WordMotionBiLSTM(
            input_size=config["model"]["input_size"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=num_classes,
            dropout=config["model"]["dropout"],
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if checkpoint.get("pipeline") != "words_motion_v1":
            raise ValueError("Checkpoint bukan berasal dari pipeline words_motion_v1")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if not args.no_speech:
            audio_player = RecordedAudioPlayer(args.audio_dir, class_names)
        else:
            print("[INFO] Pemutaran rekaman suara dinonaktifkan (--no_speech)")

        if args.server_url:
            app_publisher = MobileAppPublisher(
                server_url=args.server_url,
                room_id=args.room_id.strip() or "demo-ta",
                api_key=args.model_api_key,
            )
        else:
            print("[INFO] Sinkronisasi app nonaktif; gunakan --server_url untuk mengaktifkan")

        max_num_hands = config.get("landmark", {}).get("max_num_hands", 2)
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=config.get("landmark", {}).get("min_detection_confidence", 0.5),
            min_tracking_confidence=config.get("landmark", {}).get("min_tracking_confidence", 0.5),
        )
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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

        def lock_result(label: str, confidence: float, source: str) -> None:
            """Lock and announce one prediction result."""
            nonlocal result_label, result_confidence, result_countdown, state
            result_label = label
            result_confidence = confidence
            result_countdown = args.result_frames
            state = "RESULT"
            print(
                f"[RESULT] {source}: {result_label} "
                f"(confidence: {result_confidence * 100:.2f}%)"
            )
            if audio_player is not None:
                audio_player.play(result_label)
            if app_publisher is not None:
                app_publisher.publish_prediction(
                    result_label,
                    result_confidence,
                    source,
                )

        print("[INFO] Webcam kata aktif. Hasil akan terkunci saat voting prediksi stabil.")
        print("[INFO] Setelah hasil tampil, cukup lakukan gerakan berikutnya.")

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
                title, detail, color = (
                    "Siap mulai gerakan",
                    "Gerakkan tangan untuk membaca kata isyarat",
                    (0, 215, 255),
                )

            elif state == "RECORDING":
                buffer.append(current.copy())
                frames_since_prediction += 1
                max_buffer_size = config["preprocessing"]["max_seq_length"] * 2
                if len(buffer) > max_buffer_size:
                    buffer.pop(0)

                if present and motion < args.motion_threshold:
                    quiet_frames += 1
                elif motion >= args.motion_threshold:
                    quiet_frames = 0

                # Lakukan inferensi berkala
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
                            f"Voting stabil ({candidate_ratio * 100:.0f}% vote)",
                        )

                # Fallback jeda / tangan diam setelah gerakan
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
                    title = f"Prediksi kata: {result_label}"
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
                title = f"Prediksi kata: {result_label}"
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
                    candidate_ratio = 0.0
                    title, detail, color = (
                        "Gerakan baru terdeteksi",
                        "Merekam kata isyarat berikutnya",
                        (0, 215, 255),
                    )
                else:
                    title = "Siap untuk kata berikutnya"
                    detail = "Gerakkan tangan ke isyarat baru; tidak perlu diturunkan"
                    color = (0, 215, 255)

            if state != "RESULT" and result_label:
                result_label = ""
                result_confidence = 0.0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            if (frame.shape[1], frame.shape[0]) != (display_w, display_h):
                interpolation = cv2.INTER_LINEAR if display_w > frame.shape[1] else cv2.INTER_AREA
                frame = cv2.resize(frame, (display_w, display_h), interpolation=interpolation)

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
        if audio_player is not None:
            audio_player.close()
        if app_publisher is not None:
            app_publisher.close()
        if hands is not None:
            hands.close()
        if pose is not None:
            pose.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

