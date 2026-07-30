"""Rekam suara pengguna untuk label huruf A-Z secara terpandu.

Jalankan dari folder Model/:
    python -m src.letters.record_letter_audio
"""

import argparse
import os
import queue
import wave

import numpy as np
import sounddevice as sd

from src.letters.predict_video import resolve_input_path


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2


def list_devices() -> None:
    """Print audio devices and their input channel counts."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    if isinstance(default_input, (int, np.integer)):
        default_input = int(default_input)
    print("Perangkat audio yang tersedia:")
    for index, device in enumerate(devices):
        marker = " (default input)" if index == default_input else ""
        print(
            f"  [{index}] {device['name']} | "
            f"input={device['max_input_channels']} | "
            f"output={device['max_output_channels']}{marker}"
        )


def record_until_enter(device=None) -> np.ndarray:
    """Record mono float32 chunks until the user presses Enter."""
    chunks = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] Status input audio: {status}")
        chunks.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        input("  MEREKAM... Tekan Enter untuk berhenti. ")

    recorded = []
    while not chunks.empty():
        recorded.append(chunks.get_nowait())
    if not recorded:
        return np.empty((0, CHANNELS), dtype=np.float32)
    return np.concatenate(recorded, axis=0)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float audio in [-1, 1] to signed 16-bit PCM."""
    clipped = np.clip(audio, -1.0, 1.0)
    return np.round(clipped * np.iinfo(np.int16).max).astype(np.int16)


def save_wav(path: str, audio: np.ndarray) -> None:
    """Save mono 44.1 kHz PCM16 WAV."""
    pcm = to_pcm16(audio)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def preview(audio: np.ndarray, device=None) -> None:
    """Play and wait for a recording preview."""
    sd.play(audio, samplerate=SAMPLE_RATE, device=device)
    sd.wait()


def ask_existing(path: str, overwrite: bool) -> str:
    """Return record, skip, or quit for an existing output file."""
    if overwrite or not os.path.exists(path):
        return "record"
    while True:
        choice = input("  File sudah ada. [L]ewati, [T]impa, atau [Q]uit? ").strip().lower()
        if choice in ("l", "s", "lewati", "skip"):
            return "skip"
        if choice in ("t", "timpa", "overwrite"):
            return "record"
        if choice in ("q", "quit", "keluar"):
            return "quit"
        print("  Pilihan tidak dikenali.")


def record_letter(letter: str, output_path: str, input_device=None, output_device=None) -> str:
    """Guide one letter through recording, preview, save, retry, skip, or quit."""
    while True:
        print(f"\n[{letter}] Ucapkan dengan jelas: \"Huruf {letter}\"")
        input("  Tekan Enter untuk mulai merekam. ")
        try:
            audio = record_until_enter(input_device)
        except sd.PortAudioError as error:
            print(f"[ERROR] Mikrofon gagal digunakan: {error}")
            return "quit"

        duration = len(audio) / SAMPLE_RATE
        if len(audio) == 0 or duration < 0.15:
            print("[WARN] Rekaman kosong atau terlalu pendek. Silakan ulangi.")
            continue

        print(f"  Durasi rekaman: {duration:.2f} detik")
        try:
            preview(audio, output_device)
        except sd.PortAudioError as error:
            print(f"[WARN] Preview tidak dapat diputar: {error}")

        while True:
            choice = input("  [S]impan, [U]langi, [L]ewati, atau [Q]uit? ").strip().lower()
            if choice in ("s", "simpan", "save", ""):
                save_wav(output_path, audio)
                print(f"  [SAVED] {output_path}")
                return "saved"
            if choice in ("u", "ulangi", "retry"):
                break
            if choice in ("l", "lewati", "skip"):
                return "skip"
            if choice in ("q", "quit", "keluar"):
                return "quit"
            print("  Pilihan tidak dikenali.")


def parse_device(value: str):
    """Convert a numeric CLI device value to int, otherwise keep its name."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def main() -> int:
    parser = argparse.ArgumentParser(description="Rekam suara sendiri untuk huruf A-Z")
    parser.add_argument("--output_dir", default="assets/letters_audio", help="Folder output WAV")
    parser.add_argument("--device", default=None, help="Index/nama mikrofon input")
    parser.add_argument("--output_device", default=None, help="Index/nama perangkat preview")
    parser.add_argument("--list_devices", action="store_true", help="Tampilkan perangkat audio lalu keluar")
    parser.add_argument("--overwrite", action="store_true", help="Izinkan menimpa file yang sudah ada")
    parser.add_argument("--start_letter", default="A", choices=list(LETTERS), help="Mulai dari huruf tertentu")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    output_dir = resolve_input_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    input_device = parse_device(args.device)
    output_device = parse_device(args.output_device)
    start_index = LETTERS.index(args.start_letter)

    print("=" * 60)
    print("PEREKAM SUARA HURUF A-Z")
    print(f"Output : {output_dir}")
    print(f"Format : mono, {SAMPLE_RATE} Hz, PCM 16-bit WAV")
    print("Ucapkan \"Huruf A\", \"Huruf B\", dan seterusnya.")
    print("=" * 60)

    saved = skipped = 0
    try:
        for letter in LETTERS[start_index:]:
            output_path = os.path.join(output_dir, f"{letter}.wav")
            existing_action = ask_existing(output_path, args.overwrite)
            if existing_action == "skip":
                skipped += 1
                continue
            if existing_action == "quit":
                break

            result = record_letter(
                letter,
                output_path,
                input_device=input_device,
                output_device=output_device,
            )
            if result == "saved":
                saved += 1
            elif result == "skip":
                skipped += 1
            else:
                break
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Perekaman dihentikan.")
    except (OSError, ValueError, sd.PortAudioError) as error:
        print(f"[ERROR] Perekaman gagal: {error}")
        return 1
    finally:
        sd.stop()

    print(f"\nSelesai. Tersimpan: {saved}, dilewati: {skipped}")
    print("Jalankan script lagi dengan --start_letter untuk melanjutkan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
