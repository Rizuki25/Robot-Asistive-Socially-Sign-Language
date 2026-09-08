"""Non-blocking, opt-in response client. No automatic retries of physical actions."""
import json
import math
import threading
import time
import uuid
from urllib.request import Request, urlopen


class RobotResponseClient:
    def __init__(self, url, token, min_confidence=0.8):
        if not token:
            raise ValueError("Set AINEX_RESPONSE_TOKEN before enabling robot responses")
        self.url = url.rstrip("/") + "/respond"
        self.token = token
        self.min_confidence = min_confidence
        self._lock = threading.Lock()
        self._busy = False
        self._until = 0.0
        self._heartbeat_at = 0.0
        self._heartbeat_busy = False

    def heartbeat(self):
        """Call only after a frame has successfully passed through MediaPipe."""
        with self._lock:
            now = time.monotonic()
            if self._heartbeat_busy or now - self._heartbeat_at < 2.0:
                return
            self._heartbeat_at = now
            self._heartbeat_busy = True
        threading.Thread(target=self._send_heartbeat, daemon=True).start()

    def _send_heartbeat(self):
        try:
            request = Request(
                self.url.rsplit("/", 1)[0] + "/readiness",
                data=json.dumps({"component": "sign", "ready": True}).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.token}, method="POST")
            with urlopen(request, timeout=2) as response:
                response.read(4096)
        except Exception as error:
            print("[ROBOT] Readiness heartbeat gagal: {}".format(type(error).__name__))
        finally:
            with self._lock:
                self._heartbeat_busy = False

    @property
    def busy(self):
        with self._lock:
            return self._busy or time.monotonic() < self._until

    def submit(self, label, confidence):
        if label not in ("Halo", "Baik"):
            return False
        if not math.isfinite(confidence) or confidence < self.min_confidence:
            print("[ROBOT] Respons ditahan: confidence belum cukup.")
            return False
        with self._lock:
            if self._busy or time.monotonic() < self._until:
                return False
            self._busy = True
        payload = {"event_id": uuid.uuid4().hex, "label": label,
                   "confidence": float(confidence)}
        threading.Thread(target=self._send, args=(payload,), daemon=True).start()
        return True

    def _send(self, payload):
        cooldown = 2.0
        try:
            request = Request(
                self.url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.token}, method="POST")
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read(4096))
            print("[ROBOT] {}: {}".format(payload["label"], result["status"]))
        except Exception as error:
            # A timeout cannot tell us whether the robot already moved.
            cooldown = 30.0
            print("[ROBOT] Status tidak pasti/gagal ({}); tidak dikirim ulang. "
                  "Periksa terminal robot.".format(type(error).__name__))
        finally:
            with self._lock:
                self._until = time.monotonic() + cooldown
                self._busy = False
