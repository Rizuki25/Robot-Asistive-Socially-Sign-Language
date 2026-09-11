import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.result_recording import ResultRecorder


class ResultRecordingTests(unittest.TestCase):
    def test_records_signs_and_status_changes_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            with patch("src.common.result_recording.time.monotonic", return_value=10) as clock:
                recorder = ResultRecorder(path)
                recorder.sign("Halo", 0.94)
                recorder.emotion({"status": "detected", "emotions": ["happy"], "confidence": 0.8})
                clock.return_value = 10.1
                recorder.emotion({"status": "detected", "emotions": ["happy"], "confidence": 0.9})
                recorder.emotion({"status": "no_face"})
                recorder.close()
                recorder.close()
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertFalse(events[0]["simulation"])
            self.assertEqual([e["type"] for e in events], ["metadata", "sign", "emotion", "emotion", "end"])
            self.assertEqual(events[1]["text"], "Halo")
            self.assertEqual(events[3]["status"], "no_face")
            self.assertAlmostEqual(events[3]["time"], 0.1)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                ResultRecorder(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
