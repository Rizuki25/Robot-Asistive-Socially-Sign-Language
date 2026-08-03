"""Perekam otomatis video dataset kata dengan kamera.

Setiap video direkam dengan jumlah frame tetap, diverifikasi, lalu disimpan ke
``dataset/words/raw/{kelas}/``. Jalankan dari folder ``Model``:

    python -m src.words.auto_recorder

Gunakan ``--label`` untuk langsung memilih kelas, misalnya:

    python -m src.words.auto_recorder --label Halo
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


MODEL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = MODEL_ROOT / "dataset" / "words" / "raw"
DEFAULT_TARGET_FPS = 30.0
DEFAULT_TARGET_FRAMES = 90
DEFAULT_MIN_CAPTURE_FPS = 27.0
DEFAULT_COUNTDOWN_DURATION = 2.0
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
INVALID_LABEL_CHARACTERS = set('<>:"/\\|?*')


def resolve_path(path_value: str) -> Path:
    """Resolve path absolut atau path relatif terhadap folder Model."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = MODEL_ROOT / path
    return path.resolve()


def validate_label(label: str) -> str:
    """Validasi nama kelas agar aman digunakan sebagai nama folder Windows."""
    normalized = label.strip()
    if not normalized:
        raise ValueError("nama kelas tidak boleh kosong")
    if normalized in {".", ".."}:
        raise ValueError("nama kelas tidak valid")
    if normalized.endswith((" ", ".")):
        raise ValueError("nama kelas tidak boleh diakhiri spasi atau titik")
    if any(character in INVALID_LABEL_CHARACTERS for character in normalized):
        raise ValueError(
            "nama kelas tidak boleh mengandung karakter: "
            + " ".join(sorted(INVALID_LABEL_CHARACTERS))
        )
    if any(ord(character) < 32 for character in normalized):
        raise ValueError("nama kelas mengandung karakter kontrol")
    return normalized


def class_labels(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        return []
    return sorted(
        (path.name for path in dataset_root.iterdir() if path.is_dir()),
        key=str.casefold,
    )


def video_count(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    return sum(
        1
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def canonical_label(label: str, available_labels: list[str]) -> str:
    """Gunakan kapitalisasi folder yang sudah ada jika namanya cocok."""
    label = validate_label(label)
    for existing_label in available_labels:
        if existing_label.casefold() == label.casefold():
            return existing_label
    return label


def prompt_for_label(dataset_root: Path) -> str:
    """Pilih kelas yang ada berdasarkan nomor atau masukkan kelas baru."""
    while True:
        labels = class_labels(dataset_root)
        print("\nKelas kata yang tersedia:")
        if labels:
            for index, label in enumerate(labels, start=1):
                count = video_count(dataset_root / label)
                print(f"  [{index:2d}] {label} ({count} video)")
        else:
            print("  (belum ada folder kelas)")

        choice = input(
            "Pilih nomor kelas atau ketik nama kelas baru: "
        ).strip()

        if choice.isdigit() and labels:
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(labels):
                return labels[selected_index]
            print("[ERROR] Nomor kelas tidak tersedia.")
            continue

        try:
            return canonical_label(choice, labels)
        except ValueError as error:
            print(f"[ERROR] {error}")


def verify_video(video_path: Path) -> tuple[int, float]:
    """Dekode ulang video dan kembalikan jumlah frame serta FPS tersimpan."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0, 0.0

    stored_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        success, _ = capture.read()
        if not success:
            break
        decoded_frames += 1
    capture.release()
    return decoded_frames, stored_fps


def save_verified_video(
    frames: list,
    output_path: Path,
    target_frames: int,
    target_fps: float,
    frame_size: tuple[int, int],
) -> bool:
    """Simpan video secara atomik setelah jumlah frame dan FPS terverifikasi."""
    if len(frames) != target_frames:
        print(
            f"[GAGAL] Jumlah frame tangkapan {len(frames)}, "
            f"seharusnya {target_frames}."
        )
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        str(temporary_path),
        fourcc,
        target_fps,
        frame_size,
    )
    if not writer.isOpened():
        print(f"[GAGAL] VideoWriter tidak dapat dibuat: {temporary_path}")
        return False

    try:
        for frame in frames:
            writer.write(cv2.resize(frame, frame_size))
    finally:
        writer.release()

    decoded_frames, stored_fps = verify_video(temporary_path)
    fps_is_valid = abs(stored_fps - target_fps) <= 0.1
    if decoded_frames != target_frames or not fps_is_valid:
        print(
            f"[GAGAL] Verifikasi video: {decoded_frames} frame, "
            f"{stored_fps:.3f} FPS. File tidak dimasukkan ke dataset."
        )
        if temporary_path.exists():
            temporary_path.unlink()
        return False

    os.replace(temporary_path, output_path)
    print(
        f"[OK] Video tersimpan dan terverifikasi: {output_path} | "
        f"{decoded_frames} frame @ {stored_fps:.1f} FPS"
    )
    return True


def camera_backend() -> int:
    """Gunakan DirectShow pada Windows untuk mengurangi startup delay."""
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def detect_all_cameras(max_index: int) -> list[dict]:
    """Scan index kamera dan kembalikan kamera yang dapat membaca frame."""
    available = []
    backend = camera_backend()
    for camera_index in range(max_index):
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            success, _ = capture.read()
            if success:
                available.append(
                    {
                        "index": camera_index,
                        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "backend": capture.getBackendName(),
                    }
                )
        capture.release()
    return available


def camera_info(camera_index: int) -> Optional[dict]:
    capture = cv2.VideoCapture(camera_index, camera_backend())
    if not capture.isOpened():
        capture.release()
        return None

    success, _ = capture.read()
    if not success:
        capture.release()
        return None

    info = {
        "index": camera_index,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "backend": capture.getBackendName(),
    }
    capture.release()
    return info


def select_camera(cameras: list[dict]) -> dict:
    print(f"Ditemukan {len(cameras)} kamera:\n")
    for list_index, camera in enumerate(cameras):
        label = "EKSTERNAL (kemungkinan)" if camera["index"] > 0 else "BUILT-IN"
        print(
            f"  [{list_index}] Index {camera['index']} - "
            f"{camera['backend']} - {camera['width']}x{camera['height']} - {label}"
        )

    if len(cameras) == 1:
        return cameras[0]

    external_cameras = [camera for camera in cameras if camera["index"] > 0]
    default_camera = external_cameras[0] if external_cameras else cameras[0]
    choice = input(
        f"Pilih nomor kamera [default {cameras.index(default_camera)}]: "
    ).strip()
    if choice.isdigit() and 0 <= int(choice) < len(cameras):
        return cameras[int(choice)]
    return default_camera


def next_label(
    current_label: str,
    available_labels: list[str],
    direction: int,
) -> str:
    if not available_labels:
        return current_label
    try:
        current_index = available_labels.index(current_label)
    except ValueError:
        return available_labels[0]
    return available_labels[(current_index + direction) % len(available_labels)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rekam video dataset kata dengan jumlah frame tetap"
    )
    parser.add_argument(
        "--dataset_dir",
        default=str(DEFAULT_DATASET_ROOT),
        help="Folder dataset raw kata (default: dataset/words/raw)",
    )
    parser.add_argument(
        "--label",
        help="Nama kelas kata; jika kosong akan dipilih secara interaktif",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        help="Index kamera; jika kosong program akan mendeteksi kamera",
    )
    parser.add_argument(
        "--max_camera_index",
        type=int,
        default=10,
        help="Jumlah index kamera yang dipindai (default: 10)",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help="Jumlah frame setiap rekaman (default: 90)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="FPS video output (default: 30)",
    )
    parser.add_argument(
        "--min_capture_fps",
        type=float,
        default=DEFAULT_MIN_CAPTURE_FPS,
        help="FPS tangkapan minimum agar video diterima (default: 27)",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=DEFAULT_COUNTDOWN_DURATION,
        help="Durasi hitung mundur sebelum merekam (default: 2 detik)",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=1440,
        help="Lebar video output (default: 1440)",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=1080,
        help="Tinggi video output (default: 1080)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_frames <= 0:
        raise SystemExit("[ERROR] --target_frames harus lebih dari 0")
    if args.target_fps <= 0:
        raise SystemExit("[ERROR] --target_fps harus lebih dari 0")
    if args.min_capture_fps <= 0:
        raise SystemExit("[ERROR] --min_capture_fps harus lebih dari 0")
    if args.min_capture_fps > args.target_fps:
        raise SystemExit("[ERROR] --min_capture_fps tidak boleh melebihi --target_fps")
    if args.countdown < 0:
        raise SystemExit("[ERROR] --countdown tidak boleh negatif")
    if args.output_width <= 0 or args.output_height <= 0:
        raise SystemExit("[ERROR] Ukuran output harus lebih dari 0")
    if args.max_camera_index <= 0:
        raise SystemExit("[ERROR] --max_camera_index harus lebih dari 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_root = resolve_path(args.dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    labels = class_labels(dataset_root)
    if args.label:
        try:
            current_label = canonical_label(args.label, labels)
        except ValueError as error:
            raise SystemExit(f"[ERROR] {error}") from error
    else:
        current_label = prompt_for_label(dataset_root)

    (dataset_root / current_label).mkdir(parents=True, exist_ok=True)
    labels = class_labels(dataset_root)

    print("\n" + "=" * 60)
    print("PEREKAM VIDEO DATASET KATA")
    print(f"Dataset : {dataset_root}")
    print(f"Kelas   : {current_label}")
    print(f"Target  : {args.target_frames} frame @ {args.target_fps:g} FPS")
    print("=" * 60)

    if args.camera_index is not None:
        selected_camera = camera_info(args.camera_index)
        if selected_camera is None:
            raise SystemExit(
                f"[ERROR] Kamera index {args.camera_index} tidak dapat diakses."
            )
    else:
        print("Mendeteksi kamera yang tersedia...")
        cameras = detect_all_cameras(args.max_camera_index)
        if not cameras:
            raise SystemExit("[ERROR] Tidak ada kamera yang dapat diakses.")
        selected_camera = select_camera(cameras)

    print(
        f"\nMenggunakan kamera index {selected_camera['index']} "
        f"({selected_camera['width']}x{selected_camera['height']})"
    )

    capture = cv2.VideoCapture(selected_camera["index"], camera_backend())
    if not capture.isOpened():
        raise SystemExit("[ERROR] Gagal membuka kamera yang dipilih.")

    capture.set(cv2.CAP_PROP_FPS, args.target_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    print(
        f"Kamera melaporkan {reported_fps:.2f} FPS. "
        f"Minimum yang diterima: {args.min_capture_fps:.1f} FPS."
    )
    print("\nKontrol:")
    print("  SPACE : mulai countdown dan merekam")
    print("  [ / ] : kelas sebelumnya / berikutnya")
    print("  L     : pilih atau buat kelas melalui terminal")
    print("  ESC   : keluar")

    state = "WAITING"
    countdown_start = 0.0
    capture_start = 0.0
    recorded_frames = []
    output_path = None
    recording_label = current_label
    frame_size = (args.output_width, args.output_height)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("[ERROR] Frame kamera tidak dapat dibaca.")
                break

            current_time = time.perf_counter()
            display_frame = frame.copy()
            overlay_x = max(30, frame.shape[1] - 500)
            cv2.putText(
                display_frame,
                f"Kelas: {current_label}",
                (overlay_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            if state == "WAITING":
                count = video_count(dataset_root / current_label)
                cv2.putText(
                    display_frame,
                    "SPACE: Rekam | [ ]: Ganti kelas | L: Pilih kelas",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Jumlah video kelas ini: {count}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )
            elif state == "COUNTDOWN":
                remaining = args.countdown - (current_time - countdown_start)
                if remaining <= 0:
                    state = "RECORDING"
                    recording_label = current_label
                    save_dir = dataset_root / recording_label
                    save_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    output_path = save_dir / (
                        f"video_{recording_label}_{timestamp}.avi"
                    )
                    recorded_frames = []
                    capture_start = time.perf_counter()
                else:
                    cv2.putText(
                        display_frame,
                        f"Mulai dalam: {int(remaining) + 1}",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 165, 255),
                        3,
                    )
            elif state == "RECORDING":
                recorded_frames.append(frame.copy())
                elapsed = max(time.perf_counter() - capture_start, 1e-6)
                capture_fps = len(recorded_frames) / elapsed
                cv2.putText(
                    display_frame,
                    "MEREKAM...",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Frame: {len(recorded_frames)}/{args.target_frames} | "
                    f"Capture: {capture_fps:.1f} FPS",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                if len(recorded_frames) >= args.target_frames:
                    state = "WAITING"
                    actual_duration = max(
                        time.perf_counter() - capture_start,
                        1e-6,
                    )
                    actual_capture_fps = len(recorded_frames) / actual_duration

                    if actual_capture_fps < args.min_capture_fps:
                        print(
                            f"[DITOLAK] Kamera hanya menangkap "
                            f"{actual_capture_fps:.2f} FPS "
                            f"(minimum {args.min_capture_fps:.1f} FPS). "
                            "Periksa pencahayaan/resolusi lalu rekam ulang."
                        )
                    elif output_path is not None:
                        saved = save_verified_video(
                            recorded_frames,
                            output_path,
                            args.target_frames,
                            args.target_fps,
                            frame_size,
                        )
                        if saved:
                            print(
                                f"Kelas: {recording_label} | "
                                f"Capture aktual: {actual_capture_fps:.2f} FPS "
                                f"selama {actual_duration:.2f} detik."
                            )
                    recorded_frames = []

            cv2.imshow("Perekam Dataset Kata", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == 32 and state == "WAITING":
                state = "COUNTDOWN"
                countdown_start = time.perf_counter()
            elif key == ord("[") and state == "WAITING":
                labels = class_labels(dataset_root)
                current_label = next_label(current_label, labels, -1)
                print(f"Kelas aktif: {current_label}")
            elif key == ord("]") and state == "WAITING":
                labels = class_labels(dataset_root)
                current_label = next_label(current_label, labels, 1)
                print(f"Kelas aktif: {current_label}")
            elif key in {ord("l"), ord("L")} and state == "WAITING":
                print("\nPindahkan fokus ke terminal untuk memilih kelas.")
                current_label = prompt_for_label(dataset_root)
                (dataset_root / current_label).mkdir(parents=True, exist_ok=True)
                labels = class_labels(dataset_root)
                print(f"Kelas aktif: {current_label}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
