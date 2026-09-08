import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

path = Path(__file__).resolve().parents[1] / 'ainex_sign_response/scripts/ready_audio_node.py'
spec = importlib.util.spec_from_file_location('ready_audio_node', path)
node = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {name: MagicMock() for name in
                              ('rospy', 'sensor_msgs', 'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg')}):
    spec.loader.exec_module(node)


class ReadyTests(unittest.TestCase):
    def test_waits_for_all_and_announces_once(self):
        gate = node.ReadyGate()
        gate.update('camera', True, 10)
        gate.update('sign', True, 10)
        self.assertFalse(gate.claim(11))
        gate.update('expression', True, 11)
        self.assertTrue(gate.claim(11))
        self.assertFalse(gate.claim(12))

    def test_stale_or_false_is_not_ready(self):
        gate = node.ReadyGate()
        for name in ('camera', 'sign', 'expression'):
            gate.update(name, True, 10)
        self.assertFalse(gate.claim(17))
        for name in ('camera', 'sign', 'expression'):
            gate.update(name, True, 18)
        gate.update('expression', False, 18)
        self.assertFalse(gate.claim(18))


if __name__ == '__main__':
    unittest.main()
