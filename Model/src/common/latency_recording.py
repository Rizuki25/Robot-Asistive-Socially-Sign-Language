"""Monotonic gesture-to-result timing, optionally persisted as a new CSV."""

import csv
import time
from pathlib import Path


class LatencyRecorder:
    fields = (
        "no", "kelas_gestur", "prediksi", "waktu_mulai_deteksi_ms",
        "waktu_prediksi_ms", "latensi_ms", "status", "confidence", "sumber",
    )

    def __init__(self, path=None, expected_label=None):
        self.origin = time.perf_counter()
        self.started = None
        self.latest = None
        self.count = 0
        self.expected_label = expected_label
        self.file = None
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("x", newline="", encoding="utf-8-sig")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
            self.writer.writeheader()
            self.file.flush()

    def start(self):
        self.started = time.perf_counter()

    def cancel(self):
        self.started = None

    def finish(self, label, confidence, source):
        ended = time.perf_counter()
        if self.started is None:
            return None
        self.count += 1
        status = "Belum dinilai"
        if self.expected_label is not None:
            status = "Sesuai" if label == self.expected_label else "Tidak sesuai"
        row = dict(zip(self.fields, (
            self.count, self.expected_label or "", label,
            round((self.started - self.origin) * 1000, 3),
            round((ended - self.origin) * 1000, 3),
            round((ended - self.started) * 1000, 3),
            status, round(float(confidence), 6), source,
        )))
        self.started = None
        self.latest = row
        if self.file is not None:
            self.writer.writerow(row)
            self.file.flush()
        return row

    def close(self):
        if self.file is not None:
            self.file.close()
