"""Optional JSONL recording for offline Manim visualization (no image data)."""
import json
import time
from pathlib import Path


class ResultRecorder:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("x", encoding="utf-8")
        self._start = time.monotonic()
        self._last_emotion = float("-inf")
        self._last_status = None
        try:
            self._write({"type": "metadata", "simulation": False, "version": 1})
        except Exception:
            self._file.close()
            raise

    def _write(self, event):
        self._file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self._file.flush()

    def sign(self, label, confidence):
        self._write({"type": "sign", "time": time.monotonic() - self._start,
                     "text": label, "confidence": float(confidence)})

    def emotion(self, result):
        now = time.monotonic()
        # Preserve status changes immediately; sample confidence at most twice/sec.
        if now - self._last_emotion < 0.5 and result["status"] == self._last_status:
            return
        self._last_emotion = now
        self._last_status = result["status"]
        self._write({**result, "type": "emotion", "time": now - self._start})

    def close(self):
        if not self._file.closed:
            try:
                self._write({"type": "end", "time": time.monotonic() - self._start})
            finally:
                self._file.close()
