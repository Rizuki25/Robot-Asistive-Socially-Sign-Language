"""
Realtime Unified Auto-Detector: Pengenalan Bahasa Isyarat (Huruf & Kata) via Webcam.
====================================================================================
Versi Kotak Compact & Cerdas:
- Jendela OpenCV kompak (rasio 4:3 / 640x480) dan overlay ramping tidak menutupi wajah.
- Arbitrase Cerdas Berbasis Rentang Gerak Fisik (Motion Span) keypoints:
  * Static Hold (motion < 0.050) -> 100% diproses sebagai HURUF (A-Z).
  * Dynamic Gesture (motion >= 0.050) -> diproses sebagai KATA (Makan, Halo, dll).
- Tombol [TAB] atau [M] untuk beralih mode manual kapan saja: AUTO -> HURUF ONLY -> KATA ONLY -> AUTO.
- Tombol [Q] atau [Esc] untuk keluar.

Jalankan dari folder Model/:
    python -m src.common.predict_webcam_auto
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

from src.common.utils import get_device, load_checkpoint
from src.letters.model import BiLSTMModel
from src.letters.preprocess import normalize_data, pad_or_truncate
from src.words.model import WordMotionBiLSTM
from src.words.preprocess import extract_word_features, pad_features

WINDOW_NAME = "Sign Language AI - Dual Auto (Huruf & Kata)"
MODEL_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def resolve_input_path(path: str) -> str:
    """Resolve input path dari direktori aktif atau root Model/."""
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(MODEL_ROOT, path)


def load_label_encoder(path: str) -> Tuple[int, list]:
    """Load label encoder JSON."""
    resolved = resolve_input_path(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Label encoder tidak ditemukan: {resolved}")
    with open(resolved, "r", encoding="utf-8") as file:
        encoder = json.load(file)
    num_classes = int(encoder["num_classes"])
    idx_to_label = encoder["idx_to_label"]
    class_names = [idx_to_label[str(idx)] for idx in range(num_classes)]
    return num_classes, class_names


def extract_webcam_landmarks(results, max_num_hands: int = 2) -> np.ndarray:
    """Ekstrak landmark 2 tangan dengan slot tetap [Left, Right]."""
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


def compute_instant_motion(previous: Optional[np.ndarray], current: np.ndarray) -> tuple[bool, float]:
    """Hitung perpindahan instan antar dua frame berurutan."""
    present = bool(np.any(current != 0))
    if previous is None:
        return present, 0.0
    valid = np.any(previous != 0, axis=1) & np.any(current != 0, axis=1)
    if not np.any(valid):
        return present, 0.0
    displacement = np.linalg.norm(current[valid, :2] - previous[valid, :2], axis=1)
    return present, float(np.mean(displacement))


def compute_comprehensive_motion(buffer: list[np.ndarray]) -> float:
    """
    Hitung rentang perpindahan spasial maksimal (motion span) dari tangan yang aktif.
    - Pada pose huruf statis: span berada di kisaran 0.015 - 0.035.
    - Pada gerakan kata dinamis: span berada di kisaran 0.065 - 0.800+.
    """
    arr = np.asarray(buffer)  # (N, 42, 3)
    valid = np.any(np.abs(arr) > 1e-8, axis=(1, 2))
    v = arr[valid]
    if len(v) < 5:
        return 0.0

    l_valid = np.any(np.abs(v[:, :21, :]) > 1e-8, axis=(1, 2))
    r_valid = np.any(np.abs(v[:, 21:, :]) > 1e-8, axis=(1, 2))

    l_span = 0.0
    if np.sum(l_valid) > 4:
        l_pts = v[l_valid, :21, :2]
        l_span = float(np.max([
            np.max(np.linalg.norm(l_pts[:, i] - np.median(l_pts[:, i], axis=0), axis=1))
            for i in range(21)
        ]))

    r_span = 0.0
    if np.sum(r_valid) > 4:
        r_pts = v[r_valid, 21:, :2]
        r_span = float(np.max([
            np.max(np.linalg.norm(r_pts[:, i] - np.median(r_pts[:, i], axis=0), axis=1))
            for i in range(21)
        ]))

    return max(l_span, r_span)


def draw_landmarks(frame: np.ndarray, frame_landmarks: np.ndarray) -> None:
    """Gambar titik landmark dan koneksi tangan."""
    height, width = frame.shape[:2]
    num_hands = frame_landmarks.shape[0] // 21
    colors = [(0, 255, 255), (255, 140, 0)]  # Kiri: Cyan, Kanan: Orange

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


def put_status(
    frame: np.ndarray,
    mode_text: str,
    title: str,
    detail: str,
    color: tuple[int, int, int],
    fps: float = 0.0,
) -> None:
    """Tampilkan overlay header status yang ramping (tidak memblokir wajah)."""
    h, w = frame.shape[:2]
    bar_height = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # Baris 1: Mode Tag + FPS + Hint
    mode_bg_color = (0, 140, 255) if mode_text == "AUTO" else (200, 100, 0)
    cv2.rectangle(frame, (12, 8), (120, 28), mode_bg_color, -1)
    cv2.putText(frame, f"MODE: {mode_text}", (18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.0f}", (130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, "[TAB]: Ganti Mode | [Q]: Keluar", (w - 210, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Baris 2: Judul Prediksi Utama
    cv2.putText(frame, title, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    # Baris 3: Detail Status
    cv2.putText(frame, detail, (14, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


class UnifiedAudioPlayer:
    """Player audio terpadu untuk huruf (A-Z.wav) dan kata (Kata.wav)."""

    def __init__(self, letters_dir: str, words_dir: str):
        self.letters_dir = resolve_input_path(letters_dir)
        self.words_dir = resolve_input_path(words_dir)
        self._missing_warnings = set()
        self.available = winsound is not None

    def _find_audio(self, category: str, label: str) -> Optional[str]:
        target_dir = self.letters_dir if category == "HURUF" else self.words_dir
        candidates = [
            os.path.join(target_dir, f"{label}.wav"),
            os.path.join(target_dir, f"{label.upper()}.wav"),
            os.path.join(target_dir, f"{label.lower()}.wav"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def play(self, category: str, label: str) -> None:
        if not self.available:
            return
        path = self._find_audio(category, label)
        if path is None:
            key = f"{category}:{label}"
            if key not in self._missing_warnings:
                print(f"[INFO] Audio {category} '{label}' belum tersedia.")
                self._missing_warnings.add(key)
            return
        try:
            winsound.PlaySound(
                path,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
        except RuntimeError as error:
            print(f"[WARN] Gagal memutar audio ({path}): {error}")

    def close(self) -> None:
        if self.available:
            winsound.PlaySound(None, 0)


class UnifiedAppPublisher:
    """Kirim hasil prediksi (baik huruf maupun kata) ke backend mobile web."""

    def __init__(self, server_url: str, room_id: str, api_key: str):
        self.server_url = server_url.rstrip("/")
        self.room_id = room_id
        self.api_key = api_key
        self._last_warning_time = 0.0
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name="unified-app-publisher",
            daemon=True,
        )
        self._thread.start()
        print(f"[INFO] Sinkronisasi App aktif: {self.server_url} (room: {self.room_id})")

    def _headers(self, content_type: str) -> dict[str, str]:
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["x-model-api-key"] = self.api_key
        return headers

    def publish(self, category: str, label: str, confidence: float, source: str) -> None:
        self._queue.put(
            {
                "roomId": self.room_id,
                "text": label,
                "category": category,
                "confidence": float(confidence),
                "source": f"predict_webcam_auto:{source}",
            }
        )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                req = urllib.request.Request(
                    f"{self.server_url}/api/sign-result",
                    data=json.dumps(item).encode("utf-8"),
                    headers=self._headers("application/json"),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    resp.read()
                print(f"[SYNC] Terkirim ke app: [{item['category']}] {item['text']} ({item['confidence'] * 100:.1f}%)")
            except (OSError, urllib.error.URLError) as err:
                now = time.monotonic()
                if now - self._last_warning_time >= 5:
                    print(f"[WARN] Sinkronisasi app gagal: {err}")
                    self._last_warning_time = now

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1)


def predict_letter_sequence(
    landmarks: np.ndarray,
    config: dict,
    model: BiLSTMModel,
    device: torch.device,
) -> Tuple[int, float]:
    """Prediksi menggunakan model Huruf."""
    normalized = normalize_data([landmarks], method=config["preprocessing"]["normalization"])
    processed = pad_or_truncate(normalized, config["preprocessing"]["max_seq_length"])
    inputs = torch.from_numpy(processed).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(inputs), dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item())


def predict_word_sequence(
    landmarks: np.ndarray,
    config: dict,
    model: WordMotionBiLSTM,
    device: torch.device,
) -> Tuple[int, float]:
    """Prediksi menggunakan model Kata (140 fitur motion-aware)."""
    features = extract_word_features(landmarks)
    data, lengths = pad_features([features], config["preprocessing"]["max_seq_length"])
    inputs = torch.from_numpy(data).to(device)
    lengths_t = torch.from_numpy(lengths).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(inputs, lengths_t), dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item())


def evaluate_auto_arbitration(
    buffer: list[np.ndarray],
    active_mode: str,
    cfg_letters: dict,
    model_letters: BiLSTMModel,
    classes_letters: list[str],
    cfg_words: dict,
    model_words: WordMotionBiLSTM,
    classes_words: list[str],
    device: torch.device,
) -> Tuple[str, str, float]:
    """
    Arbitrase Cerdas:
    - Menghitung rentang gerak spasial (motion_span).
    - Jika motion_span < 0.050 -> POSE STATIS (HURUF A-Z).
    - Jika motion_span >= 0.050 -> GERAKAN DINAMIS (KATA).
    """
    landmarks_np = np.asarray(buffer)

    # 1. Mode Manual Lock
    if active_mode == "HURUF ONLY":
        idx, conf = predict_letter_sequence(landmarks_np, cfg_letters, model_letters, device)
        return "HURUF", classes_letters[idx], conf

    if active_mode == "KATA ONLY":
        idx, conf = predict_word_sequence(landmarks_np, cfg_words, model_words, device)
        return "KATA", classes_words[idx], conf

    # 2. Mode AUTO: Cek Spasial Motion
    motion_span = compute_comprehensive_motion(buffer)

    if motion_span < 0.050:
        # Pose statis (Huruf)
        idx, conf = predict_letter_sequence(landmarks_np, cfg_letters, model_letters, device)
        return "HURUF", classes_letters[idx], conf
    else:
        # Gerakan dinamis (Kata)
        wrd_idx, wrd_conf = predict_word_sequence(landmarks_np, cfg_words, model_words, device)
        # Jika model kata ragu (< 0.50) tapi huruf sangat kuat (>= 0.90)
        if wrd_conf < 0.50:
            let_idx, let_conf = predict_letter_sequence(landmarks_np, cfg_letters, model_letters, device)
            if let_conf >= 0.90:
                return "HURUF", classes_letters[let_idx], let_conf
        return "KATA", classes_words[wrd_idx], wrd_conf


def consensus_check(
    predictions: list[Tuple[str, str, float]],
    vote_window: int,
    vote_ratio: float,
    stable_confidence: float,
) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]], float, float]:
    """Cek konsensus voting (Kategori + Label harus konsisten)."""
    if not predictions:
        return None, None, 0.0, 0.0

    items = [(cat, lbl) for cat, lbl, _ in predictions]
    counts = Counter(items)
    conf_map = {
        item: float(np.mean([conf for cat, lbl, conf in predictions if (cat, lbl) == item]))
        for item in counts
    }
    candidate = max(counts, key=lambda item: (counts[item], conf_map[item]))
    ratio = counts[candidate] / len(predictions)
    candidate_conf = conf_map[candidate]

    stable = (
        candidate
        if len(predictions) >= vote_window
        and ratio >= vote_ratio
        and candidate_conf >= stable_confidence
        else None
    )
    return stable, candidate, ratio, candidate_conf


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime Sign Language AI: Auto Detection (Huruf & Kata)")
    parser.add_argument("--camera_index", type=int, default=0, help="Index webcam (default: 0)")
    parser.add_argument("--letters_config", default="configs/letters_90.yaml", help="Path config huruf (default: configs/letters_90.yaml)")
    parser.add_argument("--letters_model", default="outputs/letters_90/models/best_model.pth", help="Path model huruf")
    parser.add_argument("--words_config", default="configs/words_motion_90.yaml", help="Path config kata (default: configs/words_motion_90.yaml)")
    parser.add_argument("--words_model", default="outputs/words_motion_90/models/best_model.pth", help="Path model kata")
    parser.add_argument("--mode", default="AUTO", choices=["AUTO", "HURUF", "KATA"], help="Mode awal (default: AUTO)")
    parser.add_argument("--display_width", type=int, default=640, help="Lebar window tampilan kotak (default: 640)")
    parser.add_argument("--display_height", type=int, default=480, help="Tinggi window tampilan kotak (default: 480)")
    parser.add_argument("--motion_threshold", type=float, default=0.001, help="Ambang gerak awal (default: 0.001)")
    parser.add_argument("--pause_frames", type=int, default=15, help="Frame jeda fallback (default: 15)")
    parser.add_argument("--result_frames", type=int, default=45, help="Durasi hasil terkunci tampil (default: 45)")
    parser.add_argument("--min_recording_frames", type=int, default=35, help="Frame minimum sebelum voting (default: 35)")
    parser.add_argument("--prediction_interval", type=int, default=3, help="Interval inferensi berkala (default: 3)")
    parser.add_argument("--vote_window", type=int, default=5, help="Ukuran window voting (default: 5)")
    parser.add_argument("--vote_ratio", type=float, default=0.8, help="Rasio vote konsensus (default: 0.8)")
    parser.add_argument("--stable_confidence", type=float, default=0.8, help="Confidence minimum kunci (default: 0.8)")
    parser.add_argument("--rearm_motion_frames", type=int, default=3, help="Frame pemicu isyarat berikutnya (default: 3)")
    parser.add_argument("--letters_audio_dir", default="assets/letters_audio", help="Folder audio huruf")
    parser.add_argument("--words_audio_dir", default="assets/words_audio", help="Folder audio kata")
    parser.add_argument("--no_speech", action="store_true", help="Nonaktifkan suara audio")
    parser.add_argument("--mirror", action="store_true", help="Cerminkan tampilan kamera")
    parser.add_argument("--server_url", default=os.environ.get("MODEL_SERVER_URL", ""), help="URL backend server")
    parser.add_argument("--room_id", default="demo-ta", help="Room backend app (default: demo-ta)")
    parser.add_argument("--model_api_key", default=os.environ.get("MODEL_API_KEY", ""), help="API key backend")
    args = parser.parse_args()

    # Mapping mode
    modes = ["AUTO", "HURUF ONLY", "KATA ONLY"]
    current_mode_idx = 0
    if args.mode == "HURUF":
        current_mode_idx = 1
    elif args.mode == "KATA":
        current_mode_idx = 2

    # Load Configs & Encoders
    device = get_device()
    print(f"\n[INFO] Menyiapkan Dual AI Model pada device: {device}")

    # 1. Model Huruf
    let_cfg_path = resolve_input_path(args.letters_config)
    with open(let_cfg_path, "r", encoding="utf-8") as f:
        cfg_letters = yaml.safe_load(f)
    num_let_classes, classes_letters = load_label_encoder(
        os.path.join(cfg_letters["paths"]["processed"], "label_encoder.json")
    )
    model_letters = BiLSTMModel(
        input_size=cfg_letters["model"]["input_size"],
        hidden_size=cfg_letters["model"]["hidden_size"],
        num_layers=cfg_letters["model"]["num_layers"],
        num_classes=num_let_classes,
        dropout=cfg_letters["model"]["dropout"],
    ).to(device)
    load_checkpoint(resolve_input_path(args.letters_model), model_letters, device=device)
    model_letters.eval()

    # 2. Model Kata
    wrd_cfg_path = resolve_input_path(args.words_config)
    with open(wrd_cfg_path, "r", encoding="utf-8") as f:
        cfg_words = yaml.safe_load(f)
    num_wrd_classes, classes_words = load_label_encoder(
        os.path.join(cfg_words["paths"]["processed"], "label_encoder.json")
    )
    model_words = WordMotionBiLSTM(
        input_size=cfg_words["model"]["input_size"],
        hidden_size=cfg_words["model"]["hidden_size"],
        num_layers=cfg_words["model"]["num_layers"],
        num_classes=num_wrd_classes,
        dropout=cfg_words["model"]["dropout"],
    ).to(device)
    wrd_ckpt = torch.load(resolve_input_path(args.words_model), map_location=device, weights_only=True)
    model_words.load_state_dict(wrd_ckpt["model_state_dict"])
    model_words.eval()

    print(f"[READY] Model Huruf ({num_let_classes} kelas) & Model Kata ({num_wrd_classes} kelas) siap!")

    # Inisialisasi Audio & Sync
    audio_player = None
    if not args.no_speech:
        audio_player = UnifiedAudioPlayer(args.letters_audio_dir, args.words_audio_dir)

    app_publisher = None
    if args.server_url:
        app_publisher = UnifiedAppPublisher(args.server_url, args.room_id, args.model_api_key)

    # Inisialisasi Kamera 640x480 (Format Kotak 4:3 Nativel)
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        print(f"[ERROR] Webcam index {args.camera_index} tidak dapat dibuka.")
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)

    ret_test, frame_test = capture.read()
    if not ret_test or frame_test is None:
        print("[ERROR] Gagal membaca frame dari webcam.")
        return 1

    display_w = args.display_width
    display_h = args.display_height

    # MediaPipe Hands (dioptimalkan)
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

    result_category = ""
    result_label = ""
    result_confidence = 0.0

    candidate_category = ""
    candidate_label = ""
    candidate_confidence = 0.0
    candidate_ratio = 0.0

    prediction_history = deque(maxlen=args.vote_window)
    frames_since_prediction = 0
    rearm_motion_buffer = []

    # FPS counter
    fps_val = 0.0
    frame_counter = 0
    fps_start_time = time.time()

    def lock_result(category: str, label: str, confidence: float, source: str) -> None:
        nonlocal result_category, result_label, result_confidence, result_countdown, state
        result_category = category
        result_label = label
        result_confidence = confidence
        result_countdown = args.result_frames
        state = "RESULT"
        print(f"[RESULT] {source} -> [{result_category}] {result_label} ({result_confidence * 100:.2f}%)")
        if audio_player is not None:
            audio_player.play(result_category, result_label)
        if app_publisher is not None:
            app_publisher.publish(result_category, result_label, result_confidence, source)

    print("\n" + "=" * 60)
    print("AI SIGN LANGUAGE DETECTION (COMPACT & OPTIMIZED)")
    print("Mode Awal :", modes[current_mode_idx])
    print("Tekan TAB / M untuk beralih mode kapan saja!")
    print("=" * 60 + "\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # Hitung FPS secara realtime
            frame_counter += 1
            if frame_counter >= 15:
                now = time.time()
                fps_val = frame_counter / (now - fps_start_time)
                fps_start_time = now
                frame_counter = 0

            # Resize frame display sesuai display_w dan display_h
            if (frame.shape[1], frame.shape[0]) != (display_w, display_h):
                frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = mp_hands.process(rgb)

            current = extract_webcam_landmarks(hand_results, max_num_hands=2)
            present, motion = compute_instant_motion(previous, current)
            previous = current.copy()

            draw_landmarks(frame, current)

            active_mode = modes[current_mode_idx]

            # ---------------- STATE MACHINE ----------------
            if state == "WAITING":
                if present and motion >= args.motion_threshold:
                    state = "RECORDING"
                    buffer = [current.copy()]
                    quiet_frames = 0
                    frames_since_prediction = 0
                    prediction_history.clear()
                    candidate_category = ""
                    candidate_label = ""
                    candidate_confidence = 0.0
                    candidate_ratio = 0.0

                title = "Siap Mulai Gerakan"
                detail = "Peragakan Huruf (diam) atau Kata (gerak)"
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

                # Inferensi Berkala
                if (
                    len(buffer) >= args.min_recording_frames
                    and frames_since_prediction >= args.prediction_interval
                ):
                    cat, lbl, conf = evaluate_auto_arbitration(
                        buffer,
                        active_mode,
                        cfg_letters,
                        model_letters,
                        classes_letters,
                        cfg_words,
                        model_words,
                        classes_words,
                        device,
                    )
                    frames_since_prediction = 0
                    prediction_history.append((cat, lbl, conf))

                    stable_item, candidate_item, candidate_ratio, candidate_confidence = consensus_check(
                        list(prediction_history),
                        args.vote_window,
                        args.vote_ratio,
                        args.stable_confidence,
                    )
                    if candidate_item is not None:
                        candidate_category, candidate_label = candidate_item
                    else:
                        candidate_category, candidate_label = "", ""

                    if stable_item is not None:
                        lock_result(
                            stable_item[0],
                            stable_item[1],
                            candidate_confidence,
                            f"Voting Konsensus ({candidate_ratio * 100:.0f}%)",
                        )

                # Fallback jeda diam
                if (
                    state == "RECORDING"
                    and len(buffer) >= args.min_recording_frames
                    and quiet_frames >= args.pause_frames
                ):
                    cat, lbl, conf = evaluate_auto_arbitration(
                        buffer,
                        active_mode,
                        cfg_letters,
                        model_letters,
                        classes_letters,
                        cfg_words,
                        model_words,
                        classes_words,
                        device,
                    )
                    lock_result(cat, lbl, conf, "Fallback Jeda Selesai")

                if state == "RESULT":
                    title = f"[{result_category}] {result_label}"
                    detail = f"Confidence: {result_confidence * 100:.1f}% | Terkunci"
                    color = (0, 255, 0)
                elif len(buffer) < args.min_recording_frames:
                    title = "Merekam Gerakan..."
                    detail = f"Mengumpulkan frame: {len(buffer)}/{args.min_recording_frames}"
                    color = (0, 215, 255)
                elif candidate_label:
                    title = f"Kandidat: [{candidate_category}] {candidate_label}"
                    detail = f"Vote: {candidate_ratio * 100:.0f}% | Conf: {candidate_confidence * 100:.1f}%"
                    color = (0, 215, 255)
                else:
                    title = "Menganalisis Isyarat..."
                    detail = "Menunggu konsensus prediksi"
                    color = (0, 215, 255)

            elif state == "RESULT":
                title = f"[{result_category}] {result_label}"
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
                    candidate_category = ""
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
                    candidate_category = ""
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
                result_category = ""
                result_label = ""
                result_confidence = 0.0

            if args.mirror:
                frame = cv2.flip(frame, 1)

            put_status(frame, active_mode, title, detail, color, fps=fps_val)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (9, ord("m"), ord("M")):  # 9 adalah ASCII tombol TAB
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                print(f"[MODE SWITCH] Beralih ke: {modes[current_mode_idx]}")

    finally:
        capture.release()
        if audio_player is not None:
            audio_player.close()
        if app_publisher is not None:
            app_publisher.close()
        mp_hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
