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

Controls:
    Left Stick          Shoulder pan & lift  (joints 0-1)  [ARM mode]
    Right Stick Y       Elbow flex           (joint 2)     [ARM mode]
    D-Pad Up/Down       Wrist flex           (joint 3)     [ARM mode]
    D-Pad Left/Right    Wrist roll           (joint 4)     [ARM mode]
    LT / RT             Gripper open/close                 [ARM mode]
    RT                  Drive forward                      [MOTOR mode]
    LT                  Drive backward                     [MOTOR mode]
    Left Stick X        Steer left/right                   [MOTOR mode]
    RB                  Switch to ARM mode
    LB                  Switch to MOTOR mode
    RB + LB             IDLE (stop everything)
    A                   Preset → HOME
    B                   Preset → MOVEMENT
    X                   Preset → DROP
    Y                   Preset → GRAB
    Start               Exit
"""

import argparse
import base64
import collections
import json
import platform
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pygame

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Image helpers (mirrors inference client)
# ──────────────────────────────────────────────────────────────

def decode_image(b64_str: str) -> np.ndarray:
    """Decode a base64 JPEG string into an HWC uint8 numpy array."""
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if CV2_AVAILABLE:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return arr


def show_images(images: dict) -> None:
    """Display decoded camera images in cv2 windows (no-op if cv2 unavailable)."""
    if not CV2_AVAILABLE:
        return
    for cam_name, img in images.items():
        if isinstance(img, np.ndarray) and img.ndim == 3:
            cv2.imshow(f"cam: {cam_name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


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
# Observability
# ──────────────────────────────────────────────────────────────

class ClientStats:
    """Rolling statistics for the control loop and network."""

    PING_INTERVAL   = 2.0   # seconds between ping measurements
    DISPLAY_INTERVAL = 1.0  # seconds between status line prints
    RTT_WINDOW      = 20    # samples kept for rolling RTT stats

    def __init__(self, control_hz: float):
        self.control_hz = control_hz
        self._target_dt = 1.0 / control_hz

        # Loop timing — gap between consecutive loop_start timestamps (includes sleep)
        # so it reflects actual achieved Hz, not just active work time.
        self._loop_times: collections.deque = collections.deque(maxlen=int(control_hz * 2))
        self._send_times: collections.deque = collections.deque(maxlen=int(control_hz * 2))
        self._last_loop_start: float | None = None
        self._late_frames = 0
        self._total_frames = 0

        # RTT / ping
        self._rtts: collections.deque = collections.deque(maxlen=self.RTT_WINDOW)
        self._last_ping_t = 0.0
        self._ping_seq = 0
        self._pending_pings: dict[int, float] = {}   # seq → send time

        # Display
        self._last_display_t = 0.0
        self._session_start = time.perf_counter()

    # ── Called each control loop iteration ──────────────────────

    def record_loop(self, loop_start: float) -> None:
        # Call with the loop_start timestamp; we compute the inter-start interval
        # ourselves so the sleep is included in the measurement.
        if self._last_loop_start is not None:
            interval = loop_start - self._last_loop_start
            self._loop_times.append(interval)
            if interval > self._target_dt * 1.5:
                self._late_frames += 1
        self._last_loop_start = loop_start
        self._total_frames += 1

    def record_send(self, send_dt: float) -> None:
        self._send_times.append(send_dt)

    # ── Ping (call from the main loop) ──────────────────────────

    def maybe_ping(self, sock: socket.socket) -> None:
        """Send a ping if PING_INTERVAL has elapsed."""
        now = time.perf_counter()
        if now - self._last_ping_t >= self.PING_INTERVAL:
            self._ping_seq += 1
            self._pending_pings[self._ping_seq] = now   # RTT measured locally, no need to send t
            send_msg(sock, {"type": "ping", "seq": self._ping_seq})
            self._last_ping_t = now

    def record_pong(self, msg: dict) -> None:
        seq = msg.get("seq")
        if seq in self._pending_pings:
            rtt_ms = (time.perf_counter() - self._pending_pings.pop(seq)) * 1000.0
            self._rtts.append(rtt_ms)

    # ── Display ─────────────────────────────────────────────────

    def maybe_print(self, current_pos: np.ndarray, enabled: bool) -> None:
        now = time.perf_counter()
        if now - self._last_display_t < self.DISPLAY_INTERVAL:
            return
        self._last_display_t = now

        # Loop Hz
        if len(self._loop_times) >= 2:
            avg_dt = np.mean(self._loop_times)
            actual_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            actual_hz = 0.0

        # Send latency
        avg_send_ms = np.mean(self._send_times) * 1000.0 if self._send_times else 0.0

        # RTT
        if self._rtts:
            rtt_avg = np.mean(self._rtts)
            rtt_min = np.min(self._rtts)
            rtt_max = np.max(self._rtts)
            rtt_jitter = np.std(self._rtts)
            rtt_str = f"{rtt_avg:.1f}ms (min={rtt_min:.1f} max={rtt_max:.1f} jitter={rtt_jitter:.1f})"
        else:
            rtt_str = "measuring…"

        late_pct = 100.0 * self._late_frames / max(self._total_frames, 1)
        uptime = now - self._session_start

        state = "ACTIVE" if enabled else "IDLE"
        pos_str = " ".join(f"{v:6.1f}" for v in current_pos)

        print(
            f"\r[{state}] "
            f"Hz={actual_hz:.1f}/{self.control_hz:.0f}  "
            f"send={avg_send_ms:.2f}ms  "
            f"rtt={rtt_str}  "
            f"late={late_pct:.1f}%  "
            f"up={uptime:.0f}s  "
            f"pos=[{pos_str}] ",
            end="", flush=True,
        )

    def print_summary(self) -> None:
        uptime = time.perf_counter() - self._session_start
        rtt_avg = np.mean(self._rtts) if self._rtts else float("nan")
        late_pct = 100.0 * self._late_frames / max(self._total_frames, 1)
        print(f"\n\n{'─'*60}")
        print(f"  Session summary")
        print(f"  Uptime:       {uptime:.1f}s")
        print(f"  Total frames: {self._total_frames}")
        print(f"  Late frames:  {self._late_frames}  ({late_pct:.1f}%)")
        print(f"  Avg RTT:      {rtt_avg:.1f} ms")
        print(f"{'─'*60}\n")


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
    "home":     np.array([0.0,   -108.0,  95.0,   55.0,   -90.0,  0.0],  dtype=np.float32),
    "movement": np.array([0.3,   -85.7,   95.0,  -23.0,   -90.0,  0.0],  dtype=np.float32),
    "grab":     np.array([-12.3,   60.9,  -38.3,  73.0,   -90.0,  64.1],  dtype=np.float32),
    # "drop":     np.array([0.7,    -3.9,  -72.2, -107.6,   -90.0, 0.0],  dtype=np.float32),
    "drop":     np.array([-2.5,   -30.0,  -98.0, -107.6,   90.0,  -3.3],  dtype=np.float32),
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
        show_images: bool = False,
        image_hz: int = 5,
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
        self.running = True

        # Mode toggle state
        # mode = None | "arm" | "motor"
        self.mode = None
        self._prev_rb = False   # previous frame button states for edge detection
        self._prev_lb = False

        # ── Camera display ──────────────────────────────────────
        self._show_images = show_images and CV2_AVAILABLE
        if show_images and not CV2_AVAILABLE:
            print("  ⚠️  --show-images requested but opencv-python is not installed — skipping")
        self._img_interval = max(1, control_frequency // max(1, image_hz))
        self._img_frame    = 0

        # ── Observability ───────────────────────────────────────
        self.stats = ClientStats(control_frequency)

        print(f"  Starting position: {np.round(self.current_position, 2)}")

    # ── Helpers ────────────────────────────────────────────────

    def apply_deadzone(self, value: float) -> float:
        if abs(value) < self.DEAD_ZONE:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.DEAD_ZONE) / (1.0 - self.DEAD_ZONE)

    def _send_position(self, position: np.ndarray) -> None:
        action_dict = {key: float(position[i]) for i, key in enumerate(JOINT_KEYS)}
        t0 = time.perf_counter()
        send_msg(self.sock, {"type": "action", "action": action_dict})
        self.stats.record_send(time.perf_counter() - t0)

    def _send_preset(self, position: np.ndarray) -> None:
        """Like _send_position but tags the message as a preset so the server
        uses two-phase execution (proximal joints first, then wrist/gripper)."""
        action_dict = {key: float(position[i]) for i, key in enumerate(JOINT_KEYS)}
        t0 = time.perf_counter()
        send_msg(self.sock, {"type": "action", "preset": True, "action": action_dict})
        self.stats.record_send(time.perf_counter() - t0)

    def _send_motor(self, motor1: float, motor2: float) -> None:
        """Send motor command. Values -1.0 to 1.0 (negative = backward, 0 = stop)."""
        send_msg(self.sock, {"type": "motor", "motor1": float(motor1), "motor2": float(motor2)})

    # ── Gamepad input (identical logic to original) ─────────────

    def get_gamepad_input(self):
        """
        Returns one of:
          np.ndarray        – arm delta action (6 floats)        [arm mode]
          dict              – motor command {"motor": True, ...}  [motor mode]
          str "preset_*"    – preset name
          str "idle"        – no mode active, nothing to send
          None              – exit requested
        """
        pygame.event.pump()

        rb = self.joystick.get_button(self.BTN_RB)
        lb = self.joystick.get_button(self.BTN_LB)

        # Rising-edge detection (tap, not hold)
        rb_pressed = rb and not self._prev_rb
        lb_pressed = lb and not self._prev_lb

        self._prev_rb = rb
        self._prev_lb = lb

        # ── Mode switching ────────────────────────────────────────
        # On bumper tap, ask the server if that mode is available.
        # Server replies with {"type": "mode_response", "ok": bool, "reason": str}
        # _handle_server_msg() picks up the reply and updates self.mode.
        if rb_pressed and lb_pressed:
            print("Mode: IDLE (both bumpers)")
            self.mode = None
            self._send_motor(0.0, 0.0)
        elif rb_pressed:
            send_msg(self.sock, {"type": "mode_request", "mode": "arm"})
        elif lb_pressed:
            send_msg(self.sock, {"type": "mode_request", "mode": "motor"})

        # ── Preset buttons (work in any mode) ────────────────────
        if self.joystick.get_button(self.BTN_START):
            self.running = False
            return None
        if self.joystick.get_button(self.BTN_A):
            print("→ Moving to HOME position")
            return "preset_home"
        if self.joystick.get_button(self.BTN_B):
            print("→ Moving to MOVEMENT position")
            return "preset_movement"
        if self.joystick.get_button(self.BTN_X):
            print("→ Moving to DROP position")
            return "preset_drop"
        if self.joystick.get_button(self.BTN_Y):
            print("→ Moving to GRAB position")
            return "preset_grab"

        # ── Motor mode ────────────────────────────────────────────
        if self.mode == "motor":
            lt = (self.joystick.get_axis(self.AXIS_LT) + 1.0) / 2.0
            rt = (self.joystick.get_axis(self.AXIS_RT) + 1.0) / 2.0
            throttle = rt - lt   # positive = forward, negative = backward
            turn = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))

            m1 = max(-1.0, min(1.0, throttle - turn))
            m2 = max(-1.0, min(1.0, throttle + turn))
            return {"motor": True, "m1": m1, "m2": m2}

        # ── Arm mode ──────────────────────────────────────────────
        if self.mode == "arm":
            left_x  = self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
            left_y  = -self.apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))
            right_y = self.apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))

            lt = (self.joystick.get_axis(self.AXIS_LT) + 1.0) / 2.0
            rt = (self.joystick.get_axis(self.AXIS_RT) + 1.0) / 2.0

            hat_x, hat_y = 0, 0
            if self.joystick.get_numhats() > 0:
                hat = self.joystick.get_hat(0)
                hat_x = -hat[0]
                hat_y = hat[1]

            action = np.zeros(len(JOINT_KEYS))
            action[0] = left_x  * self.max_speed   # shoulder_pan
            action[1] = left_y  * self.max_speed   # shoulder_lift
            action[2] = right_y * self.max_speed   # elbow_flex
            action[3] = hat_y   * self.max_speed   # wrist_flex
            action[4] = hat_x   * self.max_speed   # wrist_roll
            if rt > 0.1:
                action[5] = -rt * self.max_speed   # close gripper
            elif lt > 0.1:
                action[5] =  lt * self.max_speed   # open gripper
            return action

        # ── Idle — nothing to send ─────────────────────────────────
        return "idle"

    def _apply_mode_response(self, msg: dict) -> None:
        requested = msg.get("mode")
        ok        = msg.get("ok", False)
        reason    = msg.get("reason", "")
        if ok:
            if requested == "arm":
                self._send_motor(0.0, 0.0)   # stop motors when entering arm mode
            self.mode = requested
            print(f"Mode: {requested.upper()}")
        else:
            print(f"⚠️  {requested.upper()} mode unavailable — {reason}")

    def _handle_server_msg(self) -> None:
        """Non-blocking check for incoming server messages (pongs, mode_response, etc.)."""
        self.sock.setblocking(False)
        try:
            msg   = recv_msg(self.sock)
            mtype = msg.get("type")
            if mtype == "pong":
                self.stats.record_pong(msg)
            elif mtype == "mode_response":
                self._apply_mode_response(msg)
        except BlockingIOError:
            pass
        except Exception:
            pass
        finally:
            self.sock.setblocking(True)

    def _maybe_show_images(self) -> None:
        """Periodically request camera frames from the server and display them."""
        if not self._show_images:
            return
        self._img_frame += 1
        if self._img_frame % self._img_interval != 0:
            return
        try:
            send_msg(self.sock, {"type": "full_obs_request", "images": True})
            # Read until we get full_obs; handle any queued pong/mode_response first.
            while True:
                resp  = recv_msg(self.sock)
                rtype = resp.get("type")
                if rtype == "full_obs":
                    imgs = {}
                    for cam_name, b64 in resp.get("images", {}).items():
                        if b64:
                            imgs[cam_name] = decode_image(b64)
                    show_images(imgs)
                    break
                elif rtype == "pong":
                    self.stats.record_pong(resp)
                elif rtype == "mode_response":
                    self._apply_mode_response(resp)
                else:
                    break   # unexpected — stop waiting to avoid blocking forever
        except Exception:
            pass

    # ── Main loop ──────────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 60)
        print("SO-101 GAMEPAD CLIENT  →  Raspberry Pi")
        print("=" * 60)

        print("\nControls:")
        print("  ── MOTOR mode (tap LB to toggle) ─────────────────────")
        print("  RT:                  Drive forward")
        print("  LT:                  Drive backward")
        print("  Left Stick X:        Steer left / right (works while moving)")
        print("  ── ARM mode (tap RB to toggle) ───────────────────────")
        print("  Left Stick:          Shoulder pan & lift  (joints 0-1)")
        print("  Right Stick Y:       Elbow flex           (joint 2)")
        print("  D-Pad Up/Down:       Wrist flex           (joint 3)")
        print("  D-Pad Left/Right:    Wrist roll           (joint 4)")
        print("  LT:                  Open gripper")
        print("  RT:                  Close gripper")
        print("  ── General ───────────────────────────────────────────")
        print("  RB + LB:             Force idle (stop everything)")
        print("  A: HOME  B: MOVEMENT  X: DROP  Y: GRAB")
        print("  Start:               Exit")
        print("\nℹ️  Tap RB → ARM mode  |  Tap LB → MOTOR mode  |  Tap again → IDLE")
        print("Starting in 1 second…\n")
        time.sleep(1)

        try:
            while self.running:
                loop_start = time.perf_counter()
                self.stats.record_loop(loop_start)   # interval since last start

                # ── Ping / pong ───────────────────────────────────────
                self.stats.maybe_ping(self.sock)
                self._handle_server_msg()

                action = self.get_gamepad_input()

                if action is None:
                    break

                # ── Motor control ──────────────────────────────
                if isinstance(action, dict) and action.get("motor"):
                    self._send_motor(action["m1"], action["m2"])
                    self.stats.maybe_print(self.current_position, False)

                # ── Idle ───────────────────────────────────────
                elif isinstance(action, str) and action == "idle":
                    pass   # nothing to send

                # ── Preset handling ────────────────────────────
                elif isinstance(action, str):
                    preset_name = action.removeprefix("preset_")
                    target = PRESET_POSITIONS.get(preset_name, PRESET_POSITIONS["home"]).copy()
                    target = np.clip(target, self.joint_limits_lower, self.joint_limits_upper)
                    self._send_preset(target)
                    self.current_position = target

                # ── Arm continuous control ─────────────────────
                else:
                    target = np.clip(
                        self.current_position + action,
                        self.joint_limits_lower,
                        self.joint_limits_upper,
                    )
                    self._send_position(target)
                    self.current_position = target
                    self.stats.maybe_print(self.current_position, self.mode == "arm")

                # ── Camera display (throttled, all modes) ──────
                self._maybe_show_images()

                sleep_time = self.control_dt - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user (Ctrl+C)")
        finally:
            self.stats.print_summary()
            print("Shutting down…")
            self.cleanup()

    def cleanup(self):
        try:
            send_msg(self.sock, {"type": "disconnect"})
        except Exception:
            pass
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
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
    parser.add_argument("--show-images", action="store_true",
                        help="Display camera frames streamed from the server (requires opencv)")
    parser.add_argument("--image-hz",  type=int, default=5,
                        help="Camera display refresh rate in Hz (default: 5)")
    args = parser.parse_args()

    sock = connect_to_server(args.host, args.tcp_port)
    try:
        controller = SO101GamepadClient(
            sock=sock,
            robot_id=args.robot_id,
            max_speed=args.max_speed,
            control_frequency=args.frequency,
            show_images=args.show_images,
            image_hz=args.image_hz,
        )
        controller.run()
    finally:
        sock.close()
        print("✓ Socket closed.")


if __name__ == "__main__":
    main()