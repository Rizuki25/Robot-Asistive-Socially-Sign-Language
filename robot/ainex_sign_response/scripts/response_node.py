#!/usr/bin/env python3
"""ROS node, Python 3.8 compatible. HTTP is LAN-only; no shell execution."""
import hmac
import json
import math
import os
from pathlib import Path
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
from std_msgs.msg import String, Bool
import yaml


def nod_steps(config):
    n = config["nod"]
    keys = ("servo_id", "center", "amplitude", "min_position", "max_position",
            "duration_ms", "cycles")
    if any(type(n[k]) is not int for k in keys):
        raise ValueError("Nod settings must be integers")
    if n["servo_id"] != 24:
        raise ValueError("This adapter supports documented head tilt servo 24 only")
    if not (1 <= n["amplitude"] <= 60 and 300 <= n["duration_ms"] <= 1000
            and 1 <= n["cycles"] <= 3):
        raise ValueError("Nod amplitude/duration/cycles outside limits")
    low, high = n["min_position"], n["max_position"]
    if not 0 <= low < high <= 1000:
        raise ValueError("Invalid calibrated position limits")
    positions = [n["center"]]
    positions += [n["center"] - n["amplitude"],
                  n["center"] + n["amplitude"]] * n["cycles"]
    positions.append(n["center"])
    if any(p < low or p > high for p in positions):
        raise ValueError("Nod exceeds calibrated limits")
    return [(n["duration_ms"], [[n["servo_id"], p]]) for p in positions]


class Responder:
    def __init__(self, config, dry_run):
        self.config = config
        self.dry_run = dry_run
        self.steps = nod_steps(config)
        self.min_confidence = float(config["min_confidence"])
        self.cooldown = float(config["cooldown_s"])
        if not (0 <= self.min_confidence <= 1 and 0 <= self.cooldown <= 30):
            raise ValueError("Invalid confidence/cooldown")
        action = config["halo_action"]
        if not isinstance(action, str) or not action or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in action):
            raise ValueError("Invalid action name")
        self.manager = None
        if not dry_run:
            if config.get("hardware_verified") is not True:
                raise ValueError("Verify Pi 4B API, greet action and head positions first")
            if not (Path(config["action_dir"]) / (action + ".d6a")).is_file():
                raise ValueError("Configured greeting action does not exist")
            from ainex_kinematics.motion_manager import MotionManager
            self.manager = MotionManager(config["action_dir"])
            if not callable(getattr(self.manager, "run_action", None)) or not callable(
                    getattr(self.manager, "set_servos_position", None)):
                raise ValueError("Unsupported MotionManager API")
        self.lock = threading.Lock()
        self.seen = OrderedDict()
        self.until = 0.0
        self.faulted = False
        self.status = rospy.Publisher("~status", String, queue_size=10, latch=True)

    def report(self, event, status):
        data = dict(event, status=status, dry_run=self.dry_run)
        self.status.publish(String(data=json.dumps(data)))
        rospy.loginfo("%s", json.dumps(data))

    def respond(self, event):
        event_id = event.get("event_id")
        confidence = event.get("confidence")
        if (not isinstance(event_id, str) or not 1 <= len(event_id) <= 80
                or event.get("label") not in ("Halo", "Baik")
                or type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return 400, {"status": "invalid_request"}
        if confidence < self.min_confidence:
            return 422, {"status": "low_confidence"}
        if not self.lock.acquire(blocking=False):
            return 409, {"status": "busy"}
        try:
            if self.faulted or rospy.is_shutdown():
                return 503, {"status": "disabled_after_fault_or_shutdown"}
            if event_id in self.seen:
                return 200, {"status": "duplicate_ignored"}
            if time.monotonic() < self.until:
                return 409, {"status": "cooldown"}
            self.seen[event_id] = True
            if len(self.seen) > 256:
                self.seen.popitem(last=False)
            self.report(event, "started")
            try:
                if self.dry_run:
                    rospy.loginfo("DRY RUN: %s", self.config["halo_action"]
                                  if event["label"] == "Halo" else self.steps)
                elif event["label"] == "Halo":
                    self.manager.run_action(self.config["halo_action"])
                else:
                    for duration, positions in self.steps:
                        if rospy.is_shutdown():
                            raise RuntimeError("Shutdown during motion")
                        self.manager.set_servos_position(duration, positions)
                        time.sleep(duration / 1000.0)
                status = "dry_run_done" if self.dry_run else "command_sequence_done"
                self.report(event, status)
                return 200, {"status": status}
            except Exception as error:
                self.faulted = True
                rospy.logerr("Motion error: %s", error)
                self.report(event, "failed")
                return 500, {"status": "failed_restart_required"}
            finally:
                self.until = time.monotonic() + self.cooldown
        finally:
            self.lock.release()


def main():
    rospy.init_node("sign_response")
    token = os.environ.get("AINEX_RESPONSE_TOKEN", "")
    if len(token) < 16:
        raise ValueError("Set AINEX_RESPONSE_TOKEN (at least 16 characters)")
    default_config = Path(__file__).resolve().parent.parent / "config/responses.yaml"
    with open(rospy.get_param("~config", str(default_config)), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dry_run = rospy.get_param("~dry_run", True)
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be a boolean")
    responder = Responder(config, dry_run)
    readiness = {
        "sign": rospy.Publisher("/sign_language/ready", Bool, queue_size=1),
        "expression": rospy.Publisher("/expression/ready", Bool, queue_size=1),
    }

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def reply(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ("/respond", "/readiness"):
                self.reply(404, {"status": "not_found"})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"status": "unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048:
                    raise ValueError("Invalid size")
                event = json.loads(self.rfile.read(size))
                if not isinstance(event, dict):
                    raise ValueError("Expected object")
            except (ValueError, UnicodeError):
                self.reply(400, {"status": "invalid_request"})
                return
            if self.path == "/readiness":
                component = event.get("component")
                if not isinstance(component, str) or component not in readiness or type(event.get("ready")) is not bool:
                    self.reply(400, {"status": "invalid_readiness"})
                    return
                readiness[component].publish(Bool(data=event["ready"]))
                self.reply(200, {"status": "readiness_received"})
                return
            code, result = responder.respond(event)
            self.reply(code, result)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((rospy.get_param("~bind", "0.0.0.0"),
                                 int(rospy.get_param("~port", 8091))), Handler)
    server.timeout = 0.5
    rospy.loginfo("Response bridge port %s, dry_run=%s", server.server_port, dry_run)
    try:
        while not rospy.is_shutdown():
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
