"""Normalisasi selektif video dataset kata ke 60 frame pada 30 FPS.

Video yang sudah tepat 60 frame/30 FPS disalin tanpa re-encode. Hanya video
yang berbeda yang di-resampling. Dataset sumber tidak pernah ditimpa.

Jalankan dari folder ``Model``:

    python -m src.words.normalize_word_videos
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


MODEL_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
FPS_TOLERANCE = 0.1


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = MODEL_ROOT / path
    return path.resolve()


def inspect_video(video_path: Path) -> tuple[int, float]:
    """Dekode seluruh video untuk menghitung frame aktual."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("video tidak dapat dibuka")

    stored_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        success, _ = capture.read()
        if not success:
            break
        decoded_frames += 1
    capture.release()

    if decoded_frames == 0:
        raise RuntimeError("video tidak memiliki frame yang dapat didekode")
    return decoded_frames, stored_fps


def verify_video(
    video_path: Path,
    target_frames: int,
    target_fps: float,
) -> tuple[int, float]:
    decoded_frames, stored_fps = inspect_video(video_path)
    if decoded_frames != target_frames:
        raise RuntimeError(
            f"hasil berisi {decoded_frames} frame; target {target_frames}"
        )
    if abs(stored_fps - target_fps) > FPS_TOLERANCE:
        raise RuntimeError(
            f"hasil memiliki {stored_fps:.3f} FPS; target {target_fps:.3f}"
        )
    return decoded_frames, stored_fps


def temporary_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")


def copy_consistent_video(
    source_path: Path,
    output_path: Path,
    target_frames: int,
    target_fps: float,
) -> tuple[int, float]:
    """Salin video konsisten tanpa mengubah data encoded-nya."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_path_for(output_path)
    if temporary_path.exists():
        temporary_path.unlink()

    shutil.copy2(source_path, temporary_path)
    try:
        output_frames, output_fps = verify_video(
            temporary_path,
            target_frames,
            target_fps,
        )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    os.replace(temporary_path, output_path)
    return output_frames, output_fps


def video_fourcc(output_path: Path) -> int:
    if output_path.suffix.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def normalize_video(
    source_path: Path,
    output_path: Path,
    source_frames: int,
    target_frames: int,
    target_fps: float,
) -> tuple[int, float]:
    """Resampling frame terdekat secara merata dari awal hingga akhir."""
    selected_indices = np.rint(
        np.linspace(0, source_frames - 1, target_frames)
    ).astype(np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_path_for(output_path)
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
    writer = cv2.VideoWriter(
        str(temporary_path),
        video_fourcc(output_path),
        target_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError("VideoWriter tidak dapat dibuat")

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
        output_frames, output_fps = verify_video(
            temporary_path,
            target_frames,
            target_fps,
        )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    os.replace(temporary_path, output_path)

    return output_frames, output_fps


def write_report(rows: list[dict], output_root: Path) -> Path:
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
        description=(
            "Normalisasi selektif dataset kata menjadi 60 frame @ 30 FPS"
        )
    )
    parser.add_argument(
        "--input_dir",
        default="dataset/words/raw",
        help="Folder sumber per kelas (default: dataset/words/raw)",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/words/raw_60",
        help="Folder output baru (default: dataset/words/raw_60)",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=60,
        help="Jumlah frame target (default: 60)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help="FPS target (default: 30)",
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
    copied_count = 0
    normalized_count = 0
    skipped_count = 0
    failed_count = 0

    print("=" * 60)
    print("NORMALISASI SELEKTIF VIDEO DATASET KATA")
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
            output_path = output_root / class_dir.name / source_path.name
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
                source_frames, source_fps = inspect_video(source_path)
                is_consistent = (
                    source_frames == args.target_frames
                    and abs(source_fps - args.target_fps) <= FPS_TOLERANCE
                )

                if is_consistent:
                    output_frames, output_fps = copy_consistent_video(
                        source_path,
                        output_path,
                        args.target_frames,
                        args.target_fps,
                    )
                    action = "copied_unchanged"
                    copied_count += 1
                else:
                    output_frames, output_fps = normalize_video(
                        source_path,
                        output_path,
                        source_frames,
                        args.target_frames,
                        args.target_fps,
                    )
                    if source_frames < args.target_frames:
                        action = "upsample"
                    elif source_frames > args.target_frames:
                        action = "downsample"
                    else:
                        action = "rewrite_fps"
                    normalized_count += 1

                report_rows.append(
                    {
                        **base_row,
                        "status": "success",
                        "action": action,
                        "source_frames": source_frames,
                        "source_fps": round(source_fps, 3),
                        "output_frames": output_frames,
                        "output_fps": round(output_fps, 3),
                        "error": "",
                    }
                )
                print(
                    f"  [OK] {source_path.name}: {action} | "
                    f"{source_frames} -> {output_frames} frame"
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
    print(f"Disalin tanpa perubahan : {copied_count}")
    print(f"Dinormalisasi           : {normalized_count}")
    print(f"Dilewati                : {skipped_count}")
    print(f"Gagal                    : {failed_count}")
    print(f"Laporan                  : {report_path}")
    print("=" * 60)

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
