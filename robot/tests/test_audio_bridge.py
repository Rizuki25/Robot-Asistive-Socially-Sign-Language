"""Run from repository root: python -m unittest discover -s robot/tests -v

The fake runner keeps these tests off any real audio device.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'audio_bridge', ROOT / 'robot/ainex_sign_response/scripts/audio_bridge.py')
bridge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge_module)

TOKEN = 'test-token-123456'
DEVICE = 'plughw:CARD=Headphones,DEV=0'


class FakePlayer:
    def __init__(self):
        self.device = DEVICE
        self.played = []
        self.played_event = threading.Event()

    def play(self, path):
        self.played.append(path)
        self.played_event.set()


def quiet(*_args):
    pass


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for subdir, names in (('words_audio', ('Halo.wav', 'Baik.wav')),
                              ('letters_audio', ('A.wav', 'Halo.wav'))):
            folder = self.root / subdir
            folder.mkdir()
            for name in names:
                (folder / name).write_bytes(b'RIFF')

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_indexed_labels_resolve(self):
        library = bridge_module.AudioLibrary(self.root)
        self.assertEqual(library.resolve('Halo').parent.name, 'words_audio')
        self.assertEqual(library.resolve('A').name, 'A.wav')
        for label in ('../Halo', 'Halo/../A', 'ready', '', None, 'Tidak Ada', 'x' * 65):
            self.assertIsNone(library.resolve(label), label)
        self.assertEqual(len(library), 3)

    def test_word_file_wins_on_duplicate_label(self):
        library = bridge_module.AudioLibrary(self.root)
        self.assertEqual(library.resolve('Halo').parent.name, 'words_audio')


class PlayerTests(unittest.TestCase):
    def test_plays_with_explicit_device(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stderr=b'')

        player = bridge_module.Player(DEVICE, runner=runner, log=quiet)
        asset = Path('/audio/Halo.wav')
        player.play(asset)
        self.assertEqual(calls[0][0], ['aplay', '-q', '-D', DEVICE, str(asset)])
        self.assertEqual(calls[0][1]['timeout'], 30.0)

    def test_reports_aplay_failure(self):
        player = bridge_module.Player(
            DEVICE, runner=lambda *a, **k: SimpleNamespace(returncode=1, stderr=b'busy'),
            log=quiet)
        with self.assertRaises(RuntimeError) as error:
            player.play(Path('/audio/Halo.wav'))
        self.assertIn('busy', str(error.exception))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root / 'words_audio').mkdir()
        (root / 'words_audio/Halo.wav').write_bytes(b'RIFF')
        self.player = FakePlayer()
        self.bridge = bridge_module.Bridge(
            bridge_module.AudioLibrary(root), self.player, log=quiet)

    def tearDown(self):
        self.bridge.close()
        self.temporary.cleanup()

    def test_queues_known_label_and_rejects_unknown(self):
        self.assertEqual(self.bridge.submit('Halo'), 'queued')
        self.assertTrue(self.player.played_event.wait(2))
        self.assertEqual(self.player.played[0].name, 'Halo.wav')
        self.assertEqual(self.bridge.submit('../Halo'), 'unknown_label')


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root / 'words_audio').mkdir()
        (root / 'words_audio/Halo.wav').write_bytes(b'RIFF')
        self.player = FakePlayer()
        self.bridge = bridge_module.Bridge(
            bridge_module.AudioLibrary(root), self.player, log=quiet)
        self.server = bridge_module.ThreadingHTTPServer(
            ('127.0.0.1', 0), bridge_module.make_handler(self.bridge, TOKEN))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:{}'.format(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.bridge.close()
        self.temporary.cleanup()

    def post(self, payload, token=TOKEN, path='/speak'):
        request = Request(
            self.base + path, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + token}, method='POST')
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())

    def test_speak_and_health(self):
        self.assertEqual(self.post({'label': 'Halo'}), (200, {'status': 'queued'}))
        self.assertTrue(self.player.played_event.wait(2))
        request = Request(self.base + '/health',
                          headers={'Authorization': 'Bearer ' + TOKEN})
        with urlopen(request, timeout=2) as response:
            health = json.loads(response.read())
        self.assertEqual(health['status'], 'ok')
        self.assertEqual(health['labels'], 1)
        self.assertEqual(health['device'], DEVICE)

    def test_rejects_bad_token_and_unknown_label(self):
        with self.assertRaises(HTTPError) as error:
            self.post({'label': 'Halo'}, token='salah')
        self.assertEqual(error.exception.code, 401)
        with self.assertRaises(HTTPError) as error:
            self.post({'label': '../Halo'})
        self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            self.post({'label': 'Tidak Ada'})
        self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
