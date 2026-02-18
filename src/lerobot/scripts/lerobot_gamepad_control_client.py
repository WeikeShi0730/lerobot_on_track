#!/usr/bin/env python3
"""
lerobot_gamepad_control_client.py  –  runs on Windows (or any machine with gamepad)
============================================================================
Reads an Xbox controller exactly like the original lerobot_gamepad_control.py,
but instead of driving the robot locally it streams action dicts to the
Raspberry Pi server over TCP.

Later you can swap the gamepad logic for a policy inference loop – the
networking layer stays the same.

Usage:
    lerobot_gamepad_control_client --host <RASPBERRY_PI_IP>
    lerobot_gamepad_control_client --host 192.168.1.42 --tcp-port 5555

Controls (unchanged from original):
    Left Stick          Shoulder pan & lift  (joints 0-1)
    Right Stick Y       Elbow flex           (joint 2)
    D-Pad Up/Down       Wrist flex           (joint 3)
    D-Pad Left/Right    Wrist roll           (joint 4)
    LT                  Open gripper
    RT                  Close gripper
    RB (hold)           Enable movement (dead-man switch)
    A                   Preset → HOME
    X                   Preset → READY
    Y                   Preset → VERTICAL
    B                   Preset → HOME
    Start               Exit
"""

import argparse
import json
import platform
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pygame


# ──────────────────────────────────────────────────────────────
# Wire protocol (must match raspi_robot_server.py)
# ──────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by remote")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    length = struct.unpack("<I", header)[0]
    raw = _recv_exact(sock, length)
    return json.loads(raw.decode("utf-8"))


def send_msg(sock: socket.socket, obj: dict) -> None:
    raw = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("<I", len(raw)) + raw)


def connect_to_server(host: str, port: int, retries: int = 10) -> socket.socket:
    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"✓ Connected to server {host}:{port}")
            return sock
        except OSError as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Could not connect to {host}:{port} after {retries} attempts")


# ──────────────────────────────────────────────────────────────
# Joint / calibration constants
# ──────────────────────────────────────────────────────────────

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

PRESET_POSITIONS = {
    "home": np.array([0.0, -108.0, 95.0, 55.0, -90.0, 0.0], dtype=np.float32),
    "ready": np.array([0.0, 0.0, 0.0, 0.0, -90.0, 0.0], dtype=np.float32),
    "vertical": np.array([0.0, 0.0, -90.0, 0.0, -90.0, 0.0], dtype=np.float32),
}

SERVO_CENTER = 2048
SERVO_UNITS_PER_DEGREE = 4096 / 360.0


def load_joint_limits(robot_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load calibrated joint limits from the same calibration file the robot uses.
    Falls back to ±180° if not found (same behaviour as original script).
    The calibration file lives on the Pi, so on Windows we either:
      a) copy the file over once, or
      b) leave it at ±180° defaults – the Pi will still enforce hardware limits.
    """
    lower = np.array([-180.0] * len(JOINT_KEYS), dtype=np.float32)
    upper = np.array([180.0]  * len(JOINT_KEYS), dtype=np.float32)

    calib_path = (
        Path.home()
        / ".cache" / "huggingface" / "lerobot" / "calibration"
        / "robots" / "so_follower" / f"{robot_id}.json"
    )

    if not calib_path.exists():
        print(f"  ⚠️  Calibration file not found at {calib_path}")
        print("  Using default ±180° limits (copy calib file from Pi for tighter limits)")
        return lower, upper

    try:
        with open(calib_path) as f:
            calib = json.load(f)
        limits_found = False
        for i, key in enumerate(JOINT_KEYS):
            joint_name = key.removesuffix(".pos")
            if joint_name in calib:
                mc = calib[joint_name]
                if "range_min" in mc and "range_max" in mc:
                    lower[i] = (mc["range_min"] - SERVO_CENTER) / SERVO_UNITS_PER_DEGREE
                    upper[i] = (mc["range_max"] - SERVO_CENTER) / SERVO_UNITS_PER_DEGREE
                    limits_found = True
        if limits_found:
            print("  ✓ Calibration limits loaded:")
            for i, k in enumerate(JOINT_KEYS):
                print(f"    {k}: [{lower[i]:.1f}°, {upper[i]:.1f}°]")
        else:
            print("  ⚠️  No range_min/range_max in calibration file, using ±180°")
    except Exception as e:
        print(f"  ⚠️  Could not parse calibration file: {e}")

    return lower, upper


# ──────────────────────────────────────────────────────────────
# Gamepad controller (no robot – sends over TCP)
# ──────────────────────────────────────────────────────────────

class SO101GamepadClient:
    def __init__(
        self,
        sock: socket.socket,
        robot_id: str = "so101_follower",
        max_speed: float = 2.0,
        control_frequency: int = 30,
    ):
        self.sock = sock
        self.max_speed = max_speed
        self.control_dt = 1.0 / control_frequency

        # ── Gamepad ────────────────────────────────────────────
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected! Please connect an Xbox controller.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"✓ Gamepad connected: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}   Buttons: {self.joystick.get_numbuttons()}")

        # ── Platform-specific button/axis mapping (identical to original) ──
        system = platform.system()
        print(f"  Platform: {system}  →  ", end="")
        if system == "Linux":
            print("using Linux/Raspberry Pi mapping")
            self.AXIS_LEFT_X  = 0
            self.AXIS_LEFT_Y  = 1
            self.AXIS_RIGHT_X = 3
            self.AXIS_RIGHT_Y = 4
            self.AXIS_LT      = 2
            self.AXIS_RT      = 5
            self.BTN_A        = 0
            self.BTN_B        = 1
            self.BTN_X        = 2
            self.BTN_Y        = 3
            self.BTN_LB       = 4
            self.BTN_RB       = 5
            self.BTN_BACK     = 6
            self.BTN_START    = 7
            self.BTN_LSTICK   = 9
            self.BTN_RSTICK   = 10
        else:
            print("using Windows/macOS mapping")
            self.AXIS_LEFT_X  = 0
            self.AXIS_LEFT_Y  = 1
            self.AXIS_RIGHT_X = 2
            self.AXIS_RIGHT_Y = 3
            self.AXIS_LT      = 4
            self.AXIS_RT      = 5
            self.BTN_A        = 0
            self.BTN_B        = 1
            self.BTN_X        = 2
            self.BTN_Y        = 3
            self.BTN_LB       = 4
            self.BTN_RB       = 5
            self.BTN_BACK     = 6
            self.BTN_START    = 7
            self.BTN_LSTICK   = 8
            self.BTN_RSTICK   = 9

        self.DEAD_ZONE = 0.15

        # ── State ──────────────────────────────────────────────
        self.joint_limits_lower, self.joint_limits_upper = load_joint_limits(robot_id)
        self.current_position = PRESET_POSITIONS["home"].copy()
        self.enabled = False
        self.running = True

        print(f"  Starting position: {np.round(self.current_position, 2)}")

    # ── Helpers ────────────────────────────────────────────────

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < self.DEAD_ZONE:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEAD_ZONE) / (1.0 - self.DEAD_ZONE)

    def _send_position(self, position: np.ndarray) -> None:
        action_dict = {key: float(position[i]) for i, key in enumerate(JOINT_KEYS)}
        send_msg(self.sock, {"type": "action", "action": action_dict})

    # ── Gamepad input (identical logic to original) ─────────────

    def get_gamepad_input(self):
        """
        Returns one of:
          np.ndarray  – delta action (6 floats)
          str         – preset name ("preset_home", "preset_ready", "preset_vertical")
          None        – exit requested
        """
        pygame.event.pump()

        rb_pressed = self.joystick.get_button(self.BTN_RB)

        if self.joystick.get_button(self.BTN_START):
            self.running = False
            return None

        if self.joystick.get_button(self.BTN_A):
            print("→ Moving to HOME position")
            return "preset_home"
        if self.joystick.get_button(self.BTN_X):
            print("→ Moving to READY position")
            return "preset_ready"
        if self.joystick.get_button(self.BTN_Y):
            print("→ Moving to VERTICAL position")
            return "preset_vertical"
        if self.joystick.get_button(self.BTN_B):
            print("Resetting to HOME position…")
            return "preset_home"

        # If RB not pressed, don't move
        if not rb_pressed:
            if self.enabled:
                print("Control disabled - release RB")
                self.enabled = False
            return np.zeros(len(JOINT_KEYS))

        if not self.enabled:
            print("Control enabled - hold RB to move")
            self.enabled = True

        # Read analog sticks (apply deadzone, inversion)
        left_x = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
        left_y = -self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))
        right_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))

        # Read triggers (normalize from -1.0~1.0 to 0.0~1.0 range)
        # Xbox triggers default to -1.0 when not pressed, +1.0 when fully pressed
        lt = (self.joystick.get_axis(self.AXIS_LT) + 1.0) / 2.0
        rt = (self.joystick.get_axis(self.AXIS_RT) + 1.0) / 2.0

        # Read D-pad (hat) for wrist flex and wrist roll control
        # Hat returns (x, y) where x is left/right, y is up/down
        # x: -1 for left, 1 for right
        # y: -1 for down, 1 for up
        hat_x, hat_y = 0, 0
        if self.joystick.get_numhats() > 0:
            hat = self.joystick.get_hat(0)
            hat_x = -hat[0]
            hat_y = hat[1]

        action = np.zeros(len(JOINT_KEYS))
        action[0] = left_x * self.max_speed          # shoulder_pan
        action[1] = left_y * self.max_speed          # shoulder_lift
        action[2] = right_y * self.max_speed         # elbow_flex
        action[3] = hat_y * self.max_speed           # wrist_flex
        action[4] = hat_x * self.max_speed           # wrist_roll
        # Gripper control
        if rt > 0.1:
            action[5] = -rt * self.max_speed         # close gripper
        elif lt > 0.1:
            action[5] = lt * self.max_speed          # open gripper

        return action

    # ── Main loop ──────────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 60)
        print("SO-101 GAMEPAD CLIENT  →  Raspberry Pi")
        print("=" * 60)
        print("\nControls:")
        print("  Left Stick:          Shoulder pan & lift  (joints 0-1)")
        print("  Right Stick Y:       Elbow flex           (joint 2)")
        print("  D-Pad Up/Down:       Wrist flex           (joint 3)")
        print("  D-Pad Left/Right:    Wrist roll           (joint 4)")
        print("  LT:                  Open gripper")
        print("  RT:                  Close gripper")
        print("  RB (hold):           Enable movement (SAFETY)")
        print("  A:                   HOME  B: HOME")
        print("  X:                   READY   Y: VERTICAL")
        print("  Start:               Exit")
        print("\n⚠️  SAFETY: Hold RB button to enable manual movement!")
        print("Starting in 3 seconds…\n")
        time.sleep(3)

        try:
            while self.running:
                loop_start = time.time()

                action = self.get_gamepad_input()

                if action is None:
                    break

                # ── Preset handling ────────────────────────────
                if isinstance(action, str):
                    preset_name = action.removeprefix("preset_")
                    target = PRESET_POSITIONS.get(preset_name, PRESET_POSITIONS["home"]).copy()
                    # Clip to calibrated safe ranges
                    # This prevents sending positions outside the servo's physical limits
                    # Uses the actual calibrated min/max from your robot's calibration file
                    target = np.clip(target, self.joint_limits_lower, self.joint_limits_upper)
                    self._send_position(target)
                    self.current_position = target
                    time.sleep(0.5)
                    continue

                # ── Continuous control ────────────────────────
                # Calculate new target position (relative control)
                target = np.clip(
                    self.current_position + action,
                    self.joint_limits_lower,
                    self.joint_limits_upper,
                )
                self._send_position(target)
                # Update current position
                self.current_position = target

                # Status print ~once per second
                if int(time.time() * 30) % 30 == 0:
                    active = np.any(np.abs(action) > 0.001)
                    if active or self.enabled:
                        status = "ACTIVE" if active else "READY"
                        print(f"[{status}] Position: {np.round(self.current_position, 2)}")

                elapsed = time.time() - loop_start
                time.sleep(max(0, self.control_dt - elapsed))

        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        finally:
            print("\nShutting down…")
            self.cleanup()

    def cleanup(self):
        try:
            send_msg(self.sock, {"type": "disconnect"})
        except Exception:
            pass
        pygame.quit()
        print("✓ Cleanup complete")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Control SO-101 (on Raspberry Pi) via Xbox controller from Windows"
    )
    parser.add_argument("--host",       required=True,          help="Raspberry Pi IP address")
    parser.add_argument("--tcp-port",   type=int, default=2222, help="TCP port (default: 2222)")
    parser.add_argument("--robot-id",   default="so101_follower")
    parser.add_argument("--max-speed",  type=float, default=2.0,
                        help="Max joint speed in degrees/step (default: 2.0)")
    parser.add_argument("--frequency",  type=int, default=30,   help="Control Hz (default: 30)")
    args = parser.parse_args()

    sock = connect_to_server(args.host, args.tcp_port)
    try:
        controller = SO101GamepadClient(
            sock=sock,
            robot_id=args.robot_id,
            max_speed=args.max_speed,
            control_frequency=args.frequency,
        )
        controller.run()
    finally:
        sock.close()
        print("✓ Socket closed.")


if __name__ == "__main__":
    main()