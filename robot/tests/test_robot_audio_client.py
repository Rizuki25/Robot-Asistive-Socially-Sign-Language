"""Run from repository root: python -m unittest discover -s robot/tests -v"""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'robot_audio', ROOT / 'Model/src/common/robot_audio.py')
audio_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_module)


class ClientTests(unittest.TestCase):
    def client(self, url='http://192.168.50.2:8092', token='token-123456'):
        with patch.object(audio_module.threading, 'Thread'):
            return audio_module.RobotAudioClient(url, token)

    def test_requires_a_token(self):
        with self.assertRaises(ValueError):
            self.client(token='')

    def test_play_queues_label_without_blocking(self):
        client = self.client()
        self.assertTrue(client.play('Halo'))
        self.assertEqual(client.requests.get_nowait(), 'Halo')
        client.close()

    def test_send_posts_label_and_token(self):
        client = self.client()
        response = MagicMock()
        response.read.return_value = b'{}'
        context = MagicMock()
        context.__enter__.return_value = response
        context.__exit__.return_value = False
        with patch.object(audio_module, 'urlopen', return_value=context) as send:
            client._send('Halo')
        request = send.call_args[0][0]
        self.assertEqual(request.full_url, 'http://192.168.50.2:8092/speak')
        self.assertEqual(json.loads(request.data.decode('utf-8')), {'label': 'Halo'})
        self.assertEqual(request.headers['Authorization'], 'Bearer token-123456')
        client.close()

    def test_network_failure_is_logged_not_retried(self):
        client = self.client()
        with patch.object(audio_module, 'urlopen', side_effect=TimeoutError) as send:
            client._send('Halo')
        self.assertEqual(send.call_count, 1)
        client.close()


if __name__ == '__main__':
    unittest.main()
