"""Run from Model: python -m unittest discover -s tests -p test_emotion_recognition.py -v"""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.common.emotion_recognition import EMOTIONS_ID, EmotionWorker, emotion_label


class EmotionTests(unittest.TestCase):
    def make_worker(self):
        model = Mock(task="classify", names=dict(enumerate(EMOTIONS_ID)))
        detector = Mock()
        detector.empty.return_value = False
        with patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}), \
                patch("pathlib.Path.is_file", return_value=True), \
                patch("src.common.emotion_recognition.cv2.CascadeClassifier", return_value=detector):
            return EmotionWorker(interval=0.01)

    def test_checkpoint_order_controls_label(self):
        names = ["happy", "angry", "disgust", "fear", "neutral", "sad", "surprise"]
        label, confidence = emotion_label([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01], names)
        self.assertEqual(label, "Senang")
        self.assertEqual(confidence, 0.9)

    def test_low_confidence_is_not_neutral(self):
        label, _ = emotion_label([1 / 7] * 7, list(EMOTIONS_ID))
        self.assertEqual(label, "Belum yakin")

    def test_compound_label_and_invalid_probabilities(self):
        label, _ = emotion_label([0.01, 0.01, 0.01, 0.48, 0.01, 0.01, 0.47], list(EMOTIONS_ID))
        self.assertEqual(label, "Senang-Terkejut")
        with self.assertRaises(ValueError):
            emotion_label([float("nan")] * 7, list(EMOTIONS_ID))

    def test_latest_pending_frame_is_owned_copy(self):
        worker = self.make_worker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        worker.submit(frame)
        frame[:] = 10
        self.assertEqual(int(worker._pending[0].max()), 0)
        worker.submit(frame)
        self.assertEqual(int(worker._pending[0].max()), 10)
        worker.close()

    def test_old_result_expires(self):
        worker = self.make_worker()
        worker._result = (time.monotonic() - 2, "Senang", 0.9)
        self.assertNotIn("Senang", worker.status())
        self.assertEqual(worker.web_result(), {"status": "waiting"})
        worker.close()

    def test_web_result_preserves_compound_and_uncertain_states(self):
        worker = self.make_worker()
        worker._result = (time.monotonic(), "Senang-Terkejut", 0.48)
        self.assertEqual(worker.web_result(), {"status": "detected",
                         "emotions": ["happy", "surprise"], "confidence": 0.48})
        worker._result = (time.monotonic(), "Belum yakin", 0.3)
        self.assertEqual(worker.web_result(), {"status": "uncertain", "confidence": 0.3})
        worker._result = (time.monotonic(), "wajah tidak terdeteksi", None)
        self.assertEqual(worker.web_result(), {"status": "no_face"})
        worker.close()

    def test_no_face_replaces_previous_prediction(self):
        worker = self.make_worker()
        worker.detector.detectMultiScale.return_value = []
        worker._result = (time.monotonic(), "Senang", 0.9)
        worker.start()
        try:
            worker.submit(np.zeros((100, 100, 3), dtype=np.uint8))
            deadline = time.monotonic() + 2
            while "wajah tidak terdeteksi" not in worker.status() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIn("wajah tidak terdeteksi", worker.status())
            worker.model.predict.assert_not_called()
        finally:
            worker.close()

    def test_clear_discards_in_flight_result(self):
        worker = self.make_worker()
        entered, release = threading.Event(), threading.Event()

        def predict(frame):
            entered.set()
            release.wait(timeout=2)
            return "Senang", 0.9

        worker._predict = predict
        worker.start()
        try:
            worker.submit(np.zeros((100, 100, 3), dtype=np.uint8))
            self.assertTrue(entered.wait(timeout=2))
            worker.clear()
            release.set()
        finally:
            release.set()
            worker.close()
        self.assertIsNone(worker._result)
        self.assertFalse(worker._thread.is_alive())

    def test_worker_failure_is_visible_and_clears_result(self):
        worker = self.make_worker()
        worker._predict = Mock(side_effect=RuntimeError("test failure"))
        worker._result = (time.monotonic(), "Senang", 0.9)
        worker.start()
        worker.submit(np.zeros((100, 100, 3), dtype=np.uint8))
        worker._thread.join(timeout=2)
        worker.close()
        self.assertIn("gagal", worker.status())
        self.assertIsNone(worker._result)


if __name__ == "__main__":
    unittest.main()
