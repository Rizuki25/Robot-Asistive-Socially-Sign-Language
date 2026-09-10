"""Bounded asynchronous delivery to the mobile web app, independent of robot IO."""
import json
import threading
import time
from collections import deque
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class WebResultClient:
    def __init__(self, url, room_id="demo-ta", api_key=""):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("web_url harus berupa URL http/https backend aplikasi")
        self.url = url.rstrip("/")
        self.room_id = room_id.strip() or "demo-ta"
        self.api_key = api_key
        self._condition = threading.Condition()
        self._signs = deque(maxlen=16)
        self._emotion = None
        self._emotion_at = float("-inf")
        self._warning_at = float("-inf")
        self._closed = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def sign(self, label, confidence):
        with self._condition:
            if self._closed:
                return
            self._signs.append({"roomId": self.room_id, "text": label,
                                "confidence": float(confidence), "source": "bilstm"})
            self._condition.notify()

    def emotion(self, result):
        with self._condition:
            now = time.monotonic()
            if self._closed or now - self._emotion_at < 0.5:
                return
            self._emotion_at = now
            self._emotion = ({**result, "roomId": self.room_id}, now)
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                while not self._closed and not self._signs and self._emotion is None:
                    self._condition.wait()
                if self._closed:
                    return
                if self._signs:
                    endpoint, payload = "sign-result", self._signs.popleft()
                else:
                    endpoint = "emotion-result"
                    payload, queued_at = self._emotion
                    self._emotion = None
                    if time.monotonic() - queued_at > 1.5:
                        payload = {"roomId": self.room_id, "status": "waiting"}
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["x-model-api-key"] = self.api_key
            try:
                request = Request(self.url + "/api/" + endpoint,
                                  data=json.dumps(payload).encode("utf-8"),
                                  headers=headers, method="POST")
                with urlopen(request, timeout=1.5) as response:
                    response.read(4096)
            except Exception as error:
                # Sign events are not retried: avoid duplicate chat/TTS.
                now = time.monotonic()
                if now - self._warning_at >= 10:
                    print("[WEB] Pengiriman gagal ({}); periksa backend aplikasi.".format(
                        type(error).__name__), flush=True)
                    self._warning_at = now

    def close(self):
        with self._condition:
            self._closed = True
            self._signs.clear()
            self._emotion = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
