"""Non-blocking client for the robot audio bridge.

The label is sent, not the audio bytes: the Pi owns the WAV files, so playback
starts even when the laptop speaker is muted and stays available offline.
Failures are logged once per clip and never retried; the video loop must keep
running.
"""
import json
import queue
import threading
from urllib.request import Request, urlopen


class RobotAudioClient:
    def __init__(self, url, token, timeout=3.0, capacity=8):
        if not token:
            raise ValueError("Set AINEX_RESPONSE_TOKEN before enabling robot audio")
        self.url = url.rstrip("/") + "/speak"
        self.token = token
        self.timeout = timeout
        self.requests = queue.Queue(capacity)
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="robot-audio", daemon=True)
        self._worker.start()

    def play(self, label):
        """Queue one clip; returns False when the local queue is full."""
        if self._closed:
            return False
        try:
            self.requests.put_nowait(label)
        except queue.Full:
            print("[AUDIO] Antrean robot penuh; '{}' dilewati.".format(label))
            return False
        return True

    def _run(self):
        while True:
            label = self.requests.get()
            if label is None:
                return
            self._send(label)

    def _send(self, label):
        try:
            request = Request(
                self.url, data=json.dumps({"label": label}).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.token}, method="POST")
            with urlopen(request, timeout=self.timeout) as response:
                response.read(1024)
        except Exception as error:
            print("[AUDIO] Robot tidak memutar '{}' ({}).".format(
                label, type(error).__name__))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=1.0)
