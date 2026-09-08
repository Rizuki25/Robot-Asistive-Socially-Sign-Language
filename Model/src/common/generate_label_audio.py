"""Generate Indonesian label WAVs in a staging directory (requires internet).

Install generation-only dependencies: pip install edge-tts imageio-ffmpeg
Run from Model/: python -m src.common.generate_label_audio
Existing runtime audio is never overwritten by this generator.
"""

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

import edge_tts
import imageio_ffmpeg

MODEL_ROOT = Path(__file__).resolve().parents[2]
LETTER_SPEECH = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ["a", "bé", "cé", "dé", "é", "ef", "gé", "ha", "i", "jé",
     "ka", "el", "em", "en", "o", "pé", "ki", "er", "es", "té",
     "u", "vé", "wé", "eks", "yé", "zet"],
))


def normalize_wav(path):
    """Match short-clip loudness, with a -1 dBFS peak ceiling."""
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        raw = wav.readframes(wav.getnframes())
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    if rms == 0:
        raise ValueError(f"Silent audio: {path}")
    gain = min(32768 * 10 ** (-20 / 20) / rms, 32767 * 10 ** (-1 / 20) / peak)
    with wave.open(str(path), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *(round(s * gain) for s in samples)))


def inspect_wav(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        duration = wav.getnframes() / wav.getframerate()
        assert 0.15 < duration < 10, (path, duration)
        samples = wav.readframes(wav.getnframes())
        assert any(samples), f"Silent audio: {path}"
    return round(duration, 3)


async def generate(args):
    encoder = json.loads(args.labels.read_text(encoding="utf-8"))
    letters = set(encoder["letters_classes"])
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="tts-", dir=args.output) as temporary:
        for label in encoder["classes"]:
            if Path(label).name != label or "/" in label or "\\" in label:
                raise ValueError(f"Unsafe label: {label}")
            spoken = LETTER_SPEECH[label] if label in letters else label
            category = "letters_audio" if label in letters else "words_audio"
            target = args.output / category / f"{label}.wav"
            target.parent.mkdir(exist_ok=True)
            mp3 = Path(temporary) / "speech.mp3"
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(
                        spoken, args.voice, rate="-10%", connect_timeout=15, receive_timeout=30,
                    ).save(str(mp3))
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))
            subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-i", str(mp3), "-af",
                "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-50dB,"
                "areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-50dB,"
                "areverse,adelay=100,apad=pad_dur=0.15",
                "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target),
            ], check=True)
            normalize_wav(target)
            duration = inspect_wav(target)
            records.append({
                "label": label, "spoken_text": spoken,
                "file": f"{category}/{label}.wav", "duration_seconds": duration,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            })
            print(f"[{len(records):02}/{len(encoder['classes'])}] {label}: {duration}s", flush=True)
    manifest = {"provider": "Microsoft Edge online TTS via edge-tts",
                "voice": args.voice, "rate": "-10%", "format": "PCM WAV mono 24000 Hz 16-bit",
                "normalization": "-20 dBFS RMS target, -1 dBFS peak ceiling",
                "files": records}
    (args.output / "audio_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="id-ID-GadisNeural")
    parser.add_argument("--labels", type=Path,
                        default=MODEL_ROOT / "dataset/combined/processed_90/label_encoder.json")
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "assets/generated_audio")
    asyncio.run(generate(parser.parse_args()))
