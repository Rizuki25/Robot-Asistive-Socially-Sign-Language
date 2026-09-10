import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from src.common.web_results import WebResultClient


class WebResultsTests(unittest.TestCase):
    def test_sign_payload_and_api_key(self):
        sent = threading.Event()
        requests = []

        def transport(request, **kwargs):
            requests.append(request)
            sent.set()
            return MagicMock()

        with patch("src.common.web_results.urlopen", side_effect=transport):
            client = WebResultClient("http://localhost:3001", "my-room", "test-key")
            client.start()
            try:
                client.sign("Halo", 0.95)
                self.assertTrue(sent.wait(2))
            finally:
                client.close()
        self.assertEqual(requests[0].full_url, "http://localhost:3001/api/sign-result")
        self.assertEqual(requests[0].get_header("X-model-api-key"), "test-key")
        self.assertEqual(json.loads(requests[0].data), {
            "roomId": "my-room", "text": "Halo", "confidence": 0.95, "source": "bilstm"})

    def test_delayed_emotion_is_not_sent_as_current_prediction(self):
        sent = threading.Event()
        requests = []

        def transport(request, **kwargs):
            requests.append(request)
            sent.set()
            return MagicMock()

        with patch("src.common.web_results.urlopen", side_effect=transport):
            client = WebResultClient("http://localhost:3001")
            client._emotion = ({"status": "detected", "emotions": ["happy"], "confidence": 0.9}, time.monotonic() - 2)
            client.start()
            try:
                self.assertTrue(sent.wait(2))
            finally:
                client.close()
        self.assertEqual(requests[0].full_url, "http://localhost:3001/api/emotion-result")
        self.assertEqual(json.loads(requests[0].data), {"roomId": "demo-ta", "status": "waiting"})

    def test_http_failure_does_not_retry_sign_event(self):
        attempted = threading.Event()

        def fail(*args, **kwargs):
            attempted.set()
            raise OSError("offline")

        with patch("src.common.web_results.urlopen", side_effect=fail) as transport:
            client = WebResultClient("http://localhost:3001")
            client.start()
            try:
                client.sign("Baik", 0.9)
                self.assertTrue(attempted.wait(2))
            finally:
                client.close()
            self.assertEqual(transport.call_count, 1)


if __name__ == "__main__":
    unittest.main()
