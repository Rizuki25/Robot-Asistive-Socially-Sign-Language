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
              nod=dict(center=0.30, amplitude=0.10, min_position=0.20,
                       max_position=0.40, duration_s=0.8, cycles=2))


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
        self.assertEqual(steps[0], (0.8, 0.30))
        self.assertEqual([p for _, p in steps], [0.30, 0.20, 0.40, 0.20, 0.40, 0.30])
        self.assertTrue(all(0.20 <= position <= 0.40 for _, position in steps))
        self.assertEqual(self.responder.respond(dict(self.event, label='Baik'))[0], 200)

    def test_nod_limits(self):
        self.config['nod']['center'] = 0.39
        with self.assertRaises(ValueError):
            node.nod_steps(self.config)

    def test_nod_uses_ros_and_returns_to_user_center(self):
        self.responder.dry_run = False
        self.responder.head_pub = MagicMock()
        self.responder.head_pub.get_num_connections.return_value = 1
        self.responder.head_message = MagicMock()
        with patch.object(node.time, 'sleep'):
            self.assertEqual(self.responder.respond(dict(self.event, label='Baik'))[0], 200)
        self.assertEqual(self.responder.head_pub.publish.call_count, 6)
        self.responder.head_message.assert_called_with(position=0.30, duration=0.8)

    def test_live_label_gate(self):
        self.responder.dry_run = False
        self.responder.live_labels = ['Baik']
        self.assertEqual(self.responder.respond(self.event)[0], 403)
        self.assertFalse(self.responder.seen)

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
