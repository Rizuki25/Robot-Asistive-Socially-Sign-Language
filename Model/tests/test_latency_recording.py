import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.latency_recording import LatencyRecorder


class LatencyRecordingTests(unittest.TestCase):
    def test_elapsed_time_csv_and_duplicate_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.csv"
            with patch("src.common.latency_recording.time.perf_counter") as clock:
                clock.return_value = 100
                recorder = LatencyRecorder(path, "Halo")
                clock.return_value = 102
                recorder.start()
                clock.return_value = 103.25
                row = recorder.finish("Halo", 0.95, "Voting stabil")
                self.assertEqual(row["waktu_mulai_deteksi_ms"], 2000)
                self.assertEqual(row["waktu_prediksi_ms"], 3250)
                self.assertEqual(row["latensi_ms"], 1250)
                self.assertEqual(row["status"], "Sesuai")
                self.assertIsNone(recorder.finish("Halo", 0.95, "duplicate"))
                # Results are flushed before shutdown.
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["kelas_gestur"], "Halo")
                recorder.close()
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    LatencyRecorder(path)
                self.assertEqual(path.read_bytes(), before)

    def test_cancel_restart_and_incorrect_label(self):
        with patch("src.common.latency_recording.time.perf_counter") as clock:
            clock.return_value = 10
            recorder = LatencyRecorder(expected_label="Saya")
            recorder.start()
            recorder.cancel()
            self.assertIsNone(recorder.finish("Saya", 0.9, "cancelled"))
            clock.return_value = 20
            recorder.start()
            clock.return_value = 22
            row = recorder.finish("Kamu", 0.7, "Fallback jeda")
            self.assertEqual(row["latensi_ms"], 2000)
            self.assertEqual(row["no"], 1)
            self.assertEqual(row["status"], "Tidak sesuai")

    def test_unknown_ground_truth_is_not_marked_correct(self):
        recorder = LatencyRecorder()
        recorder.start()
        row = recorder.finish("Apa", 0.99, "Voting stabil")
        self.assertEqual(row["kelas_gestur"], "")
        self.assertEqual(row["status"], "Belum dinilai")
        recorder.close()


if __name__ == "__main__":
    unittest.main()
