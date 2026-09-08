"""Run from repository root: python -m unittest discover -s robot/tests -v

ROS and YAML are stubbed; these tests never connect to robot hardware.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'response_node', ROOT / 'robot/ainex_sign_response/scripts/response_node.py')
node = importlib.util.module_from_spec(spec)
ros = MagicMock()
ros.is_shutdown.return_value = False
with patch.dict(sys.modules, {'rospy': ros, 'std_msgs': MagicMock(),
                              'std_msgs.msg': MagicMock(), 'yaml': MagicMock()}):
    spec.loader.exec_module(node)

CONFIG = dict(hardware_verified=False, action_dir='/not/a/robot', halo_action='greet',
              min_confidence=0.8, cooldown_s=0,
              nod=dict(servo_id=24, center=500, amplitude=35, min_position=450,
                       max_position=550, duration_ms=500, cycles=2))


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(CONFIG)
        self.responder = node.Responder(self.config, True)
        self.event = dict(event_id='test-1', label='Halo', confidence=0.95)

    def test_dry_run_and_duplicate(self):
        self.assertIsNone(self.responder.manager)
        self.assertEqual(self.responder.respond(self.event)[1]['status'], 'dry_run_done')
        self.assertEqual(self.responder.respond(self.event)[1]['status'], 'duplicate_ignored')

    def test_live_requires_verified_hardware(self):
        with self.assertRaises(ValueError):
            node.Responder(self.config, False)

    def test_invalid_requests(self):
        for update in (dict(label='Makan'), dict(confidence=float('nan')),
                       dict(confidence=True), dict(event_id='')):
            self.assertEqual(self.responder.respond(dict(self.event, **update))[0], 400)
        self.assertEqual(self.responder.respond(dict(self.event, confidence=0.2))[0], 422)

    def test_busy_rejects_without_queue(self):
        with self.responder.lock:
            self.assertEqual(self.responder.respond(self.event)[0], 409)
        self.assertFalse(self.responder.seen)

    def test_nod_returns_to_center_and_only_uses_tilt(self):
        steps = node.nod_steps(self.config)
        self.assertEqual(steps[0], steps[-1])
        self.assertEqual(len(steps), 6)
        self.assertTrue(all(positions[0][0] == 24 for _, positions in steps))
        self.assertEqual(self.responder.respond(dict(self.event, label='Baik'))[0], 200)

    def test_nod_limits(self):
        self.config['nod']['center'] = 540
        with self.assertRaises(ValueError):
            node.nod_steps(self.config)

    def test_failure_disables_further_motion(self):
        self.responder.dry_run = False
        self.responder.manager = MagicMock()
        self.responder.manager.run_action.side_effect = RuntimeError('driver error')
        self.assertEqual(self.responder.respond(self.event)[0], 500)
        self.assertEqual(self.responder.respond(dict(self.event, event_id='test-2'))[0], 503)


spec_client = importlib.util.spec_from_file_location(
    'robot_response', ROOT / 'Model/src/common/robot_response.py')
client_module = importlib.util.module_from_spec(spec_client)
spec_client.loader.exec_module(client_module)


class ClientTests(unittest.TestCase):
    def test_filter_and_single_inflight(self):
        client = client_module.RobotResponseClient('http://localhost:8091', 'test-token')
        with patch.object(client_module.threading, 'Thread') as worker:
            self.assertFalse(client.submit('Makan', 0.95))
            self.assertFalse(client.submit('Halo', 0.3))
            self.assertTrue(client.submit('Halo', 0.95))
            self.assertTrue(client.busy)
            self.assertFalse(client.submit('Baik', 0.95))
            self.assertEqual(worker.call_count, 1)

    def test_network_failure_no_retry_and_cooldown(self):
        client = client_module.RobotResponseClient('http://localhost:8091', 'test-token')
        with patch.object(client_module, 'urlopen', side_effect=TimeoutError) as request:
            client._send(dict(event_id='x', label='Halo', confidence=0.9))
        self.assertEqual(request.call_count, 1)
        self.assertTrue(client.busy)


if __name__ == '__main__':
    unittest.main()
