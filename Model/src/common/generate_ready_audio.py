"""Generate the robot readiness announcement once; requires internet."""
import argparse
import asyncio
from pathlib import Path
import subprocess
import tempfile
import wave

import edge_tts
import imageio_ffmpeg

TEXT = "Halo, robot pengenalan bahasa isyarat dan ekspresi wajah sudah siap digunakan."
OUTPUT = Path(__file__).resolve().parents[3] / "robot/ainex_sign_response/audio/ready.wav"


async def generate(output):
    if output.exists():
        raise FileExistsError("Audio already exists; choose a different --output to regenerate")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ready-tts-", dir=output.parent) as temporary:
        mp3 = Path(temporary) / "ready.mp3"
        wav = Path(temporary) / "ready.wav"
        await edge_tts.Communicate(TEXT, "id-ID-GadisNeural", rate="-10%",
                                   connect_timeout=15, receive_timeout=30).save(str(mp3))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(mp3), "-ac", "1", "-ar", "24000",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        with wave.open(str(wav), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
            if not 0.5 < duration < 30 or not any(stream.readframes(stream.getnframes())):
                raise ValueError("Invalid or silent generated audio")
        wav.replace(output)
    print("Voice: id-ID-GadisNeural\nText: {}\nAudio: {}".format(TEXT, output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    asyncio.run(generate(parser.parse_args().output))
