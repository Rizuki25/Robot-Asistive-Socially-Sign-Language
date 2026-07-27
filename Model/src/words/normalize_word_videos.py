"""Normalisasi video dataset kata menjadi jumlah frame dan FPS yang seragam.

Program tidak pernah menimpa video sumber. Struktur folder kelas dipertahankan
di folder output baru. Audio tidak disalin karena tidak digunakan oleh model.

Jalankan dari folder ``Model``:

    python -m src.words.normalize_word_videos
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def resolve_path(path_value: str) -> Path:
    """Resolve path absolut atau path relatif terhadap folder Model."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = MODEL_ROOT / path
    return path.resolve()


def inspect_video(video_path: Path) -> tuple[int, float]:
    """Dekode video untuk memperoleh jumlah frame aktual dan FPS metadata."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("video tidak dapat dibuka")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        success, _ = capture.read()
        if not success:
            break
        decoded_frames += 1
    capture.release()

    if decoded_frames == 0:
        raise RuntimeError("video tidak memiliki frame yang dapat didekode")
    return decoded_frames, source_fps


def build_frame_indices(source_frames: int, target_frames: int) -> np.ndarray:
    """Pilih frame secara merata dari seluruh durasi video."""
    return np.rint(
        np.linspace(0, source_frames - 1, target_frames)
    ).astype(np.int64)


def verify_output(
    video_path: Path,
    target_frames: int,
    target_fps: float,
) -> tuple[int, float]:
    """Pastikan output dapat didekode dan memiliki frame/FPS yang diminta."""
    decoded_frames, stored_fps = inspect_video(video_path)
    if decoded_frames != target_frames:
        raise RuntimeError(
            f"hasil berisi {decoded_frames} frame; target {target_frames}"
        )
    if abs(stored_fps - target_fps) > 0.1:
        raise RuntimeError(
            f"hasil tersimpan pada {stored_fps:.3f} FPS; target {target_fps:.3f}"
        )
    return decoded_frames, stored_fps


def normalize_video(
    source_path: Path,
    output_path: Path,
    target_frames: int,
    target_fps: float,
) -> dict:
    """Normalisasi satu video dengan nearest-frame temporal resampling."""
    source_frames, source_fps = inspect_video(source_path)
    selected_indices = build_frame_indices(source_frames, target_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    if temporary_path.exists():
        temporary_path.unlink()

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError("video tidak dapat dibuka pada tahap normalisasi")

    success, frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError("frame pertama tidak dapat dibaca")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(temporary_path),
        fourcc,
        target_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError("VideoWriter MP4 tidak dapat dibuat")

    source_index = 0
    target_index = 0
    try:
        try:
            while success and target_index < target_frames:
                while (
                    target_index < target_frames
                    and selected_indices[target_index] == source_index
                ):
                    writer.write(frame)
                    target_index += 1

                source_index += 1
                success, frame = capture.read()
        finally:
            capture.release()
            writer.release()
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    if target_index != target_frames:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            f"hanya {target_index}/{target_frames} frame yang berhasil ditulis"
        )

    try:
        output_frames, output_fps = verify_output(
            temporary_path,
            target_frames,
            target_fps,
        )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    os.replace(temporary_path, output_path)

    if source_frames < target_frames:
        action = "upsample"
    elif source_frames > target_frames:
        action = "downsample"
    else:
        action = "rewrite_fps"

    return {
        "status": "success",
        "action": action,
        "source_frames": source_frames,
        "source_fps": round(source_fps, 3),
        "output_frames": output_frames,
        "output_fps": round(output_fps, 3),
        "error": "",
    }


def write_report(rows: list[dict], output_root: Path) -> Path:
    """Tulis laporan setiap video secara atomik."""
    report_path = output_root / "normalization_report.csv"
    temporary_path = output_root / "normalization_report.tmp.csv"
    fieldnames = [
        "class",
        "source",
        "output",
        "status",
        "action",
        "source_frames",
        "source_fps",
        "output_frames",
        "output_fps",
        "error",
    ]

    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalisasi video dataset kata menjadi tepat 90 frame @ 30 FPS"
    )
    parser.add_argument(
        "--input_dir",
        default="dataset/words/raw",
        help="Folder video sumber per kelas (default: dataset/words/raw)",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/words/raw_90",
        help="Folder output baru (default: dataset/words/raw_90)",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=90,
        help="Jumlah frame output setiap video (default: 90)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help="FPS output setiap video (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_dir)

    if args.target_frames <= 0:
        raise SystemExit("[ERROR] --target_frames harus lebih dari 0")
    if args.target_fps <= 0:
        raise SystemExit("[ERROR] --target_fps harus lebih dari 0")
    if not input_root.is_dir():
        raise SystemExit(f"[ERROR] Folder input tidak ditemukan: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise SystemExit(
            "[ERROR] Folder output harus terpisah dan tidak boleh berada "
            "di dalam folder input."
        )

    class_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(f"[ERROR] Tidak ada folder kelas di: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    report_rows = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    print("=" * 60)
    print("NORMALISASI VIDEO DATASET KATA")
    print(f"Input         : {input_root}")
    print(f"Output baru   : {output_root}")
    print(f"Target        : {args.target_frames} frame @ {args.target_fps:g} FPS")
    print("Video sumber  : tidak akan ditimpa")
    print("=" * 60)

    for class_dir in class_dirs:
        video_files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        print(f"\n[{class_dir.name}] {len(video_files)} video")

        for source_path in video_files:
            output_path = output_root / class_dir.name / f"{source_path.stem}.mp4"
            base_row = {
                "class": class_dir.name,
                "source": str(source_path),
                "output": str(output_path),
            }

            if output_path.exists():
                skipped_count += 1
                report_rows.append(
                    {
                        **base_row,
                        "status": "skipped",
                        "action": "output_exists",
                        "source_frames": "",
                        "source_fps": "",
                        "output_frames": "",
                        "output_fps": "",
                        "error": "output sudah ada dan tidak ditimpa",
                    }
                )
                print(f"  [SKIP] {source_path.name}: output sudah ada")
                continue

            try:
                result = normalize_video(
                    source_path,
                    output_path,
                    args.target_frames,
                    args.target_fps,
                )
                success_count += 1
                report_rows.append({**base_row, **result})
                print(
                    f"  [OK] {source_path.name}: "
                    f"{result['source_frames']} -> {result['output_frames']} frame"
                )
            except Exception as error:
                failed_count += 1
                report_rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "action": "",
                        "source_frames": "",
                        "source_fps": "",
                        "output_frames": "",
                        "output_fps": "",
                        "error": str(error),
                    }
                )
                print(f"  [GAGAL] {source_path.name}: {error}")

    report_path = write_report(report_rows, output_root)
    print("\n" + "=" * 60)
    print("SELESAI")
    print(f"Berhasil : {success_count}")
    print(f"Dilewati : {skipped_count}")
    print(f"Gagal    : {failed_count}")
    print(f"Laporan  : {report_path}")
    print("=" * 60)

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
