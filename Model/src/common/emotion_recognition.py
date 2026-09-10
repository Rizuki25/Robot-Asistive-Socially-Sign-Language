"""Optional face emotion worker; receives frames from the sign camera only."""
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = Path(__file__).resolve().parents[3] / (
    "Emotion/webcam/models/fer2013_baseline-2/weights/best.pt"
)
EMOTIONS_ID = {
    "angry": "Marah", "disgust": "Jijik", "fear": "Takut",
    "happy": "Senang", "neutral": "Netral", "sad": "Sedih",
    "surprise": "Terkejut",
}


def emotion_label(probabilities, names):
    """Use checkpoint class order and the existing visual decision thresholds."""
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != (len(names),)
            or not np.all(np.isfinite(probabilities))):
        raise ValueError("Probabilitas model ekspresi tidak valid")
    first, second = np.argsort(probabilities)[::-1][:2]
    confidence = float(probabilities[first])
    if confidence < 0.40:
        return "Belum yakin", confidence
    label = EMOTIONS_ID[names[first]]
    if confidence - float(probabilities[second]) < 0.20:
        label += "-" + EMOTIONS_ID[names[second]]
    return label, confidence


class EmotionWorker:
    """One pending latest frame, bounded inference rate, expiring results."""

    def __init__(self, model_path=DEFAULT_MODEL, device="cpu", interval=0.2):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("emotion_interval harus positif dan finite")
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError("Model ekspresi tidak ditemukan: {}".format(model_path))
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        if self.model.task != "classify":
            raise ValueError("Model ekspresi harus bertipe classify")
        names = self.model.names
        self.names = [names[i] for i in range(len(names))]
        if len(self.names) != 7 or set(self.names) != set(EMOTIONS_ID):
            raise ValueError("Kelas model ekspresi tidak sesuai: {}".format(self.names))
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.detector.empty():
            raise RuntimeError("Haar Cascade wajah gagal dimuat")
        self.device = device
        self.interval = interval
        self.max_age = 1.5
        self._condition = threading.Condition()
        self._pending = None
        self._result = None
        self._generation = 0
        self._stopped = False
        self._error = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._condition:
            if not self._stopped and self._error is None:
                # Own the raw image before the main thread draws hand overlays.
                self._pending = (frame.copy(), time.monotonic(), self._generation)
                self._condition.notify()

    def clear(self):
        """Discard pending and in-flight results when robot motion pauses inference."""
        with self._condition:
            self._generation += 1
            self._pending = None
            self._result = None

    def status(self):
        with self._condition:
            if self._error is not None:
                return "Ekspresi: gagal; lihat terminal"
            if self._result is None:
                return "Ekspresi: menunggu pemrosesan wajah"
            captured_at, label, confidence = self._result
            if time.monotonic() - captured_at > self.max_age:
                return "Ekspresi: menunggu hasil baru"
            if confidence is None:
                return "Ekspresi: " + label
            return "Ekspresi: {} ({:.0f}%)".format(label, confidence * 100)

    def web_result(self):
        """Structured, fresh output for the web UI; never reuse an expired label."""
        with self._condition:
            if self._error is not None:
                return {"status": "error"}
            if self._result is None:
                return {"status": "waiting"}
            captured_at, label, confidence = self._result
            if time.monotonic() - captured_at > self.max_age:
                return {"status": "waiting"}
            if confidence is None:
                return {"status": "no_face"}
            if label == "Belum yakin":
                return {"status": "uncertain", "confidence": confidence}
            reverse = {value: key for key, value in EMOTIONS_ID.items()}
            return {"status": "detected", "confidence": confidence,
                    "emotions": [reverse[part] for part in label.split("-")]}

    def _predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return "wajah tidak terdeteksi", None
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        px, py = int(w * 0.20), int(h * 0.20)
        crop = frame[max(0, y - py):min(frame.shape[0], y + h + py),
                     max(0, x - px):min(frame.shape[1], x + w + px)]
        results = self.model.predict(
            crop, imgsz=224, device=self.device, verbose=False
        )
        return emotion_label(results[0].probs.data.cpu().numpy(), self.names)

    def _run(self):
        next_at = 0.0
        try:
            while True:
                with self._condition:
                    while not self._stopped:
                        delay = next_at - time.monotonic()
                        if self._pending is not None and delay <= 0:
                            break
                        self._condition.wait(timeout=max(delay, 0.001)
                                             if self._pending is not None else None)
                    if self._stopped:
                        return
                    frame, captured_at, generation = self._pending
                    self._pending = None
                next_at = time.monotonic() + self.interval
                label, confidence = self._predict(frame)
                with self._condition:
                    if not self._stopped and generation == self._generation:
                        self._result = (captured_at, label, confidence)
        except Exception as error:
            with self._condition:
                self._error = str(error)
                self._pending = None
                self._result = None
            print("[EMOTION] Pemrosesan gagal: {}".format(error), flush=True)

    def close(self):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)


def draw_emotion_status(frame, text):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 30), (width, height), (25, 25, 25), -1)
    cv2.putText(frame, text, (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 230, 160), 1, cv2.LINE_AA)
