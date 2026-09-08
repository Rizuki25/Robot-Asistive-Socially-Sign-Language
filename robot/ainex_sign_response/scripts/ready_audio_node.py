#!/usr/bin/env python3
"""Announce once after fresh camera, sign pipeline and expression heartbeats."""
from pathlib import Path
import subprocess
import threading
import time

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class ReadyGate:
    def __init__(self, timeout=6.0):
        self.timeout = timeout
        self.times = {}
        self.claimed = False
        self.lock = threading.Lock()

    def update(self, component, ready, now):
        with self.lock:
            if ready:
                self.times[component] = now
            else:
                self.times.pop(component, None)

    def claim(self, now):
        with self.lock:
            fresh = all(name in self.times and 0 <= now - self.times[name] < self.timeout
                        for name in ("camera", "sign", "expression"))
            if self.claimed or not fresh:
                return False
            self.claimed = True
            return True


def main():
    rospy.init_node("ready_audio")
    default = Path(__file__).resolve().parent.parent / "audio/ready.wav"
    audio = Path(rospy.get_param("~audio_file", str(default)))
    if not audio.is_file():
        raise FileNotFoundError("Generate and copy ready.wav first: {}".format(audio))
    device = rospy.get_param("~audio_device", "plughw:CARD=Headphones,DEV=0")
    gate = ReadyGate()
    subscriptions = [
        rospy.Subscriber("/camera/image_raw", Image,
                         lambda msg: gate.update("camera", True, time.monotonic()),
                         queue_size=1, buff_size=4 * 1024 * 1024),
        rospy.Subscriber("/sign_language/ready", Bool,
                         lambda msg: gate.update("sign", msg.data, time.monotonic()), queue_size=1),
        rospy.Subscriber("/expression/ready", Bool,
                         lambda msg: gate.update("expression", msg.data, time.monotonic()), queue_size=1),
    ]
    rospy.loginfo("Waiting for camera + sign + expression readiness; audio device=%s", device)
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        if gate.claim(time.monotonic()):
            try:
                subprocess.run(["aplay", "-q", "-D", device, str(audio)],
                               check=True, timeout=35)
                rospy.loginfo("Ready announcement playback completed")
            except (OSError, subprocess.SubprocessError) as error:
                rospy.logerr("Ready announcement failed: %s. Check audio device and restart node.", error)
        rate.sleep()
    for subscription in subscriptions:
        subscription.unregister()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
