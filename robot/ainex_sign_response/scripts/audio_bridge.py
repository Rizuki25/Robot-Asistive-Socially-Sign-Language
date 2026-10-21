#!/usr/bin/env python3
"""Play label WAVs on the robot speaker over the LAN.

Runs on the AiNex Raspberry Pi and does not need ROS. The laptop posts a label
name; the bridge only accepts names that exist in a fixed index built from the
audio folders, so a request can never select an arbitrary path. Playback is
serialized on one worker thread to keep two clips from fighting over the ALSA
device.

    export AINEX_RESPONSE_TOKEN='token-yang-sama-dengan-laptop'
    python3 audio_bridge.py --device plughw:CARD=Headphones,DEV=0
"""
import argparse
import hmac
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"
DEFAULT_DEVICE = "plughw:CARD=Headphones,DEV=0"
AUDIO_SUBDIRS = ("words_audio", "letters_audio")
MAX_BODY = 1024


def _log(message):
    print(message, flush=True)


class AudioLibrary:
    """Fixed label index; names outside the index are rejected."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = {}
        for subdir in AUDIO_SUBDIRS:
            for path in sorted((self.root / subdir).glob("*.wav")):
                self.files.setdefault(path.stem, path)

    def resolve(self, label):
        if not isinstance(label, str) or not 1 <= len(label) <= 64:
            return None
        return self.files.get(label)

    def __len__(self):
        return len(self.files)


class Player:
    """One aplay process per clip; no shell, no user-controlled arguments."""

    def __init__(self, device=DEFAULT_DEVICE, timeout=30.0, runner=None, log=_log):
        self.device = device
        self.timeout = timeout
        self.runner = runner or subprocess.run
        self.log = log

    def play(self, path):
        command = ["aplay", "-q", "-D", self.device, str(path)]
        self.log("[AUDIO] memutar {} di {}".format(path.name, self.device))
        try:
            result = self.runner(
                command, timeout=self.timeout,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("aplay melebihi {} detik".format(self.timeout))
        except OSError as error:
            raise RuntimeError("aplay tidak bisa dijalankan: {}".format(error))
        if result.returncode != 0:
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError("aplay keluar {}: {}".format(result.returncode, detail))


class Bridge:
    """Bounded FIFO so a slow device cannot grow memory without limit."""

    def __init__(self, library, player, capacity=8, log=_log):
        self.library = library
        self.player = player
        self.log = log
        self.queue = queue.Queue(capacity)
        self.playing = False
        self.closed = False
        self.worker = threading.Thread(target=self._run, name="audio-bridge", daemon=True)
        self.worker.start()

    def submit(self, label):
        path = self.library.resolve(label)
        if path is None:
            return "unknown_label"
        try:
            self.queue.put_nowait(path)
        except queue.Full:
            return "busy"
        return "queued"

    def _run(self):
        while True:
            path = self.queue.get()
            if path is None:
                return
            self.playing = True
            try:
                self.player.play(path)
            except Exception as error:  # playback must not kill the worker
                self.log("[AUDIO] gagal: {}".format(error))
            finally:
                self.playing = False

    def status(self):
        return {"queued": self.queue.qsize(), "playing": self.playing}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=2.0)


def make_handler(bridge, token):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ainex-audio/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def reply(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            provided = self.headers.get("Authorization", "").encode("utf-8", "replace")
            return hmac.compare_digest(provided, ("Bearer " + token).encode("utf-8"))

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"status": "not_found"})
                return
            if not self.authorized():
                self.reply(401, {"status": "unauthorized"})
                return
            payload = {"status": "ok", "device": bridge.player.device,
                       "labels": len(bridge.library)}
            payload.update(bridge.status())
            self.reply(200, payload)

        def do_POST(self):
            if self.path != "/speak":
                self.reply(404, {"status": "not_found"})
                return
            if not self.authorized():
                self.reply(401, {"status": "unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("invalid size")
                event = json.loads(self.rfile.read(size))
                if not isinstance(event, dict):
                    raise ValueError("expected object")
            except (ValueError, UnicodeError):
                self.reply(400, {"status": "invalid_request"})
                return
            status = bridge.submit(event.get("label"))
            code = {"queued": 200, "unknown_label": 404, "busy": 429}[status]
            self.reply(code, {"status": status})

        def log_message(self, *args):
            pass

    return Handler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=Path, default=DEFAULT_AUDIO_DIR,
                        help="Folder berisi words_audio/ dan letters_audio/")
    parser.add_argument("--device", default=os.environ.get("AINEX_AUDIO_DEVICE", DEFAULT_DEVICE),
                        help="Perangkat ALSA, contoh plughw:CARD=Headphones,DEV=0")
    parser.add_argument("--bind", default="0.0.0.0", help="Alamat bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8092, help="Port HTTP (default: 8092)")
    parser.add_argument("--queue", type=int, default=8, help="Batas antrean klip (default: 8)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    token = os.environ.get("AINEX_RESPONSE_TOKEN", "")
    if len(token) < 16:
        raise SystemExit("Set AINEX_RESPONSE_TOKEN (minimal 16 karakter)")
    if shutil.which("aplay") is None:
        raise SystemExit("aplay tidak ditemukan; pasang alsa-utils di robot")
    library = AudioLibrary(args.audio_dir)
    if not len(library):
        raise SystemExit("Tidak ada WAV di {}/words_audio atau letters_audio".format(args.audio_dir))
    bridge = Bridge(library, Player(args.device), max(1, args.queue))
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(bridge, token))
    _log("[AUDIO] {} label, device {}, port {}".format(
        len(library), args.device, server.server_port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
