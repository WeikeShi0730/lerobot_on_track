#!/usr/bin/env python3
"""
lerobot_gamepad_control_server.py  –  runs on server Raspberry Pi
================================================
Receives action dicts from the Windows gamepad client (or policy client)
over TCP and drives the SO-101 follower arm.

Usage:
    # Both arm and motors (default)
    python3 lerobot_gamepad_control_server.py

    # Arm only
    python3 lerobot_gamepad_control_server.py --no-motors

    # Motors only
    python3 lerobot_gamepad_control_server.py --no-arm

    # Custom serial port or TCP port
    python3 lerobot_gamepad_control_server.py --port /dev/ttyACM1 --tcp-port 5555

Wire protocol (shared with client):
    Each message = [4-byte little-endian uint32 length][UTF-8 JSON payload]
"""

import argparse
import collections
import json
import socket
import struct
import time
from pathlib import Path

import numpy as np

# Arm control (lerobot SO-101)
try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    ARM_AVAILABLE = True
except Exception as _arm_import_err:
    print(f"⚠️  lerobot import failed: {_arm_import_err}  — arm control disabled")
    SO101Follower = None
    SO101FollowerConfig = None
    ARM_AVAILABLE = False

# Motor control (gpiozero + lgpio for Raspberry Pi 5)
try:
    from gpiozero import Motor, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    Device.pin_factory = LGPIOFactory()
    MOTORS_AVAILABLE = True
except Exception as _gpio_err:
    print(f"⚠️  GPIO motor init failed: {_gpio_err}  — motor control disabled")
    MOTORS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Wire protocol helpers
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


# ──────────────────────────────────────────────────────────────
# Observability
# ──────────────────────────────────────────────────────────────

class ServerStats:
    """Per-session receive and execution metrics."""

    DISPLAY_INTERVAL = 2.0   # seconds between log lines
    HZ_WINDOW        = 60    # samples for rolling Hz estimate

    def __init__(self):
        self._action_times: collections.deque = collections.deque(maxlen=self.HZ_WINDOW)
        self._exec_times:   collections.deque = collections.deque(maxlen=self.HZ_WINDOW)
        self._last_action_t: float | None = None
        self._last_display_t = time.perf_counter()
        self._session_start  = time.perf_counter()
        self._total_actions  = 0
        self._total_pings    = 0

    def record_action(self, recv_t: float, exec_s: float) -> None:
        # recv_t = timestamp taken BEFORE robot.send_action() — marks when the
        #          message arrived, independent of how long execution took.
        # exec_s = how long robot.send_action() itself took (servo comms).
        if self._last_action_t is not None:
            self._action_times.append(recv_t - self._last_action_t)
        self._last_action_t = recv_t
        self._exec_times.append(exec_s)
        self._total_actions += 1

    def record_ping(self) -> None:
        self._total_pings += 1

    def maybe_print(self) -> None:
        now = time.perf_counter()
        if now - self._last_display_t < self.DISPLAY_INTERVAL:
            return
        self._last_display_t = now

        if len(self._action_times) >= 2:
            avg_inter = np.mean(self._action_times)
            recv_hz   = 1.0 / avg_inter if avg_inter > 0 else 0.0
            jitter_ms = np.std(self._action_times) * 1000.0
        else:
            recv_hz   = 0.0
            jitter_ms = 0.0

        exec_ms = np.mean(self._exec_times) * 1000.0 if self._exec_times else 0.0
        uptime   = now - self._session_start

        print(
            f"[server] recv={recv_hz:.1f}Hz  "
            f"exec={exec_ms:.2f}ms  "
            f"jitter={jitter_ms:.2f}ms  "
            f"actions={self._total_actions}  "
            f"pings={self._total_pings}  "
            f"up={uptime:.0f}s"
        )

    def print_summary(self) -> None:
        uptime   = time.perf_counter() - self._session_start
        exec_avg = np.mean(self._exec_times) * 1000.0 if self._exec_times else float("nan")
        print(f"\n[server] ── Session summary ──────────────────────────")
        print(f"[server]   Uptime:        {uptime:.1f}s")
        print(f"[server]   Total actions: {self._total_actions}")
        print(f"[server]   Total pings:   {self._total_pings}")
        print(f"[server]   Avg exec time: {exec_avg:.2f} ms")
        print(f"[server] ──────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────
# Motor controller
# ──────────────────────────────────────────────────────────────

class MotorController:
    """
    Wraps two gpiozero Motor objects.
    Receives speed values in [-1.0, 1.0]:
      positive → forward, negative → backward, 0 → stop
    """

    def __init__(
        self,
        # Motor 1 pins
        m1_forward: int = 5,
        m1_backward: int = 6,
        m1_enable: int = 13,
        # Motor 2 pins
        m2_forward: int = 16,
        m2_backward: int = 1,
        m2_enable: int = 12,
    ):
        if not MOTORS_AVAILABLE:
            raise RuntimeError("gpiozero/lgpio not available — cannot create MotorController")
        self.motor1 = Motor(forward=m1_forward, backward=m1_backward, enable=m1_enable, pwm=True)
        self.motor2 = Motor(forward=m2_forward, backward=m2_backward, enable=m2_enable, pwm=True)
        print(f"✓ Motors initialised  M1=(fwd={m1_forward},bwd={m1_backward},en={m1_enable})  "
              f"M2=(fwd={m2_forward},bwd={m2_backward},en={m2_enable})")

    def set(self, m1: float, m2: float) -> None:
        """
        Set motor speeds. Values in [-1.0, 1.0].
        Positive = forward, negative = backward, 0 = stop (coast).
        """
        self._drive(self.motor1, m1)
        self._drive(self.motor2, m2)

    @staticmethod
    def _drive(motor, value: float) -> None:
        value = max(-1.0, min(1.0, value)) # clamp
        if abs(value) < 0.01:
            motor.stop()
        elif value > 0:
            motor.forward(value)
        else:
            motor.backward(-value)

    def stop(self) -> None:
        self.motor1.stop()
        self.motor2.stop()


# ──────────────────────────────────────────────────────────────
# Robot helpers
# ──────────────────────────────────────────────────────────────

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# For preset moves, drive proximal joints first (shoulder/elbow) so the arm
# is in a safe configuration before the wrist and gripper move.
# Continuous control sends all joints together as usual.
PRESET_PHASE1 = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos"}
PRESET_PHASE2 = {"wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}
PRESET_PHASE_PAUSE = 0.5   # seconds to wait between phase 1 and phase 2

PRESET_SPEED_DEG_PER_SEC = 50.0   # max joint speed during preset interpolation
PRESET_STEP_HZ           = 50     # interpolation tick rate (steps per second)


def send_action_two_phase(robot, action_dict: dict) -> float:
    """
    Execute a preset action in two phases to avoid collision, with smooth
    interpolation so no joint moves faster than PRESET_SPEED_DEG_PER_SEC.

      Phase 1 — shoulder_pan, shoulder_lift, elbow_flex  (interpolated)
      Phase 2 — wrist_flex, wrist_roll, gripper          (interpolated, after pause)

    Returns total wall time taken (for stats).
    """
    t0 = time.perf_counter()

    def interpolate_phase(phase_keys: set) -> None:
        phase_target = {k: v for k, v in action_dict.items() if k in phase_keys}
        if not phase_target:
            return

        # Read current joint positions from the robot.
        try:
            obs = robot.get_observation()
            phase_start = {k: float(obs[k]) for k in phase_target if k in obs}
        except Exception:
            phase_start = {}

        # Fall back to jumping straight to target if observation unavailable.
        if not phase_start:
            robot.send_action(phase_target)
            return

        # Number of steps driven by the largest joint displacement.
        step_dt   = 1.0 / PRESET_STEP_HZ
        max_delta = max(abs(phase_target[k] - phase_start[k]) for k in phase_target)
        min_steps = max(1, int(np.ceil(max_delta / (PRESET_SPEED_DEG_PER_SEC * step_dt))))

        for i in range(1, min_steps + 1):
            alpha = i / min_steps
            step_cmd = {
                k: phase_start[k] + alpha * (phase_target[k] - phase_start[k])
                for k in phase_target
            }
            robot.send_action(step_cmd)
            time.sleep(step_dt)

    phase1_keys = {k for k in action_dict if k in PRESET_PHASE1}
    phase2_keys = {k for k in action_dict if k in PRESET_PHASE2}

    interpolate_phase(phase1_keys)
    if phase1_keys and phase2_keys:
        time.sleep(PRESET_PHASE_PAUSE)
    interpolate_phase(phase2_keys)

    return time.perf_counter() - t0


def connect_robot_arm(port: str, robot_id: str):
    """Connect to SO-101 arm. Returns robot instance or None on failure."""
    if not ARM_AVAILABLE:
        print("ℹ️  lerobot not available — arm disabled")
        return None
    try:
        print(f"  Connecting to SO-101 on {port} (id={robot_id}) …")
        config = SO101FollowerConfig(port=port, id=robot_id)
        robot = SO101Follower(config)
        robot.connect()
        print("✓ Arm connected.")
        return robot
    except Exception as e:
        print(f"⚠️  Could not connect to arm: {e}  — arm control disabled")
        return None


def get_observation_dict(robot) -> dict:
    """Return all scalar observations as a plain Python dict (JSON-serialisable)."""
    obs = robot.get_observation()
    return {
        k: float(v) if hasattr(v, "item") else v
        for k, v in obs.items()
        if not k.startswith("observation.image")
    }


# ──────────────────────────────────────────────────────────────
# Client handler
# ──────────────────────────────────────────────────────────────

def handle_client(conn: socket.socket, robot_arm, motors: "MotorController | None", verbose: bool) -> None:
    stats = ServerStats()
    try:
        while True:
            msg = recv_msg(conn)
            mtype = msg.get("type")

            # ── mode_request ─────────────────────────────────────────────
            # Client tapped RB or LB — check if that mode is available and reply
            if mtype == "mode_request":
                requested = msg.get("mode")
                if requested == "arm":
                    ok     = robot_arm is not None
                    reason = "arm not connected" if not ok else ""
                elif requested == "motor":
                    ok     = motors is not None
                    reason = "motors not connected" if not ok else ""
                else:
                    ok     = False
                    reason = f"unknown mode '{requested}'"
                send_msg(conn, {"type": "mode_response", "mode": requested, "ok": ok, "reason": reason})
                print(f"[server] mode_request '{requested}' → {'granted' if ok else f'denied ({reason})'}")

            # ── action (arm) ─────────────────────────────────────────────
            elif mtype == "action":
                if robot_arm is not None:
                    action_dict: dict = msg["action"]   # {joint_key: float, …}
                    is_preset: bool   = msg.get("preset", False)
                    recv_t = time.perf_counter()
                    t0     = time.perf_counter()
                    if is_preset:
                        # Phase 1: shoulder_pan, shoulder_lift, elbow_flex
                        # Phase 2: wrist_flex, wrist_roll, gripper
                        # Pause between phases lets the arm clear collisions.
                        send_action_two_phase(robot_arm, action_dict)
                    else:
                        robot_arm.send_action(action_dict)
                    exec_s = time.perf_counter() - t0
                    stats.record_action(recv_t, exec_s)
                    stats.maybe_print()
                    if verbose:
                        kind = "preset" if is_preset else "continuous"
                        vals = [f"{action_dict.get(k, 0):.1f}" for k in JOINT_KEYS]
                        print(f"[server] action({kind}) exec={exec_s*1000:.2f}ms  pos={vals}")
                else:
                    if verbose:
                        print("[server] action msg received but arm not connected — ignoring")

            # ── motor ────────────────────────────────────────────────────
            elif mtype == "motor":
                if motors is not None:
                    m1 = float(msg.get("motor1", 0.0))
                    m2 = float(msg.get("motor2", 0.0))
                    motors.set(m1, m2)
                    if verbose:
                        print(f"[server] motor  m1={m1:+.2f}  m2={m2:+.2f}")
                else:
                    if verbose:
                        print("[server] motor msg received but motors not connected — ignoring")

            # ── ping → pong ─────────────────────────────────────────────
            elif mtype == "ping":
                stats.record_ping()
                send_msg(conn, {"type": "pong", "seq": msg.get("seq")})

            # ── observation request ─────────────────────────────────────
            elif mtype == "obs_request":
                if robot_arm is not None:
                    obs = get_observation_dict(robot_arm)
                    send_msg(conn, {"type": "obs", "data": obs})
                else:
                    send_msg(conn, {"type": "obs", "data": {}, "error": "arm not connected"})

            # ── graceful disconnect ─────────────────────────────────────
            elif mtype == "disconnect":
                print("[server] Client requested disconnect.")
                break

            else:
                print(f"[server] Unknown message type: {mtype!r}")

    except ConnectionError as e:
        print(f"[server] Connection lost: {e}")
    finally:
        if motors is not None:
            motors.stop()
            print("[server] Motors stopped.")
        stats.print_summary()
        conn.close()
        print("[server] Client connection closed.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SO-101 + motor TCP server (Raspberry Pi)\n"
                    "Run with arm only, motors only, or both — any combination is valid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # TCP
    parser.add_argument("--host",      default="0.0.0.0",      help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--tcp-port",  type=int, default=2222,  help="TCP port (default: 2222)")
    parser.add_argument("--verbose",   action="store_true",     help="Print every action/motor command")
    # Arm
    parser.add_argument("--no-arm",    action="store_true",     help="Disable arm (motors only)")
    parser.add_argument("--port",      default="/dev/ttyACM0",  help="Serial port for SO-101 (default: /dev/ttyACM0)")
    parser.add_argument("--robot-id",  default="so101_follower")
    # Motors
    parser.add_argument("--no-motors", action="store_true",     help="Disable motors (arm only)")
    # parser.add_argument("--m1-fwd",    type=int, default=17,    help="Motor 1 forward GPIO / IN1 (default: 17)")
    # parser.add_argument("--m1-bwd",    type=int, default=18,    help="Motor 1 backward GPIO / IN2 (default: 18)")
    # parser.add_argument("--m1-en",     type=int, default=25,    help="Motor 1 enable GPIO / ENA (default: 25)")
    # parser.add_argument("--m2-fwd",    type=int, default=22,    help="Motor 2 forward GPIO / IN3 (default: 22)")
    # parser.add_argument("--m2-bwd",    type=int, default=23,    help="Motor 2 backward GPIO / IN4 (default: 23)")
    # parser.add_argument("--m2-en",     type=int, default=24,    help="Motor 2 enable GPIO / ENB (default: 24)")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  SO-101 + Motor TCP Server")
    print("=" * 55)

    # ── Arm ───────────────────────────────────────────────────
    robot_arm = None
    if args.no_arm:
        print("ℹ️  Arm:    DISABLED (--no-arm)")
    else:
        robot_arm = connect_robot_arm(args.port, args.robot_id)
        if robot_arm is None:
            print("ℹ️  Arm:    DISABLED (connection failed)")
        else:
            print(f"✓  Arm:    ENABLED  ({args.port})")

    # ── Motors ────────────────────────────────────────────────
    motors = None
    if args.no_motors:
        print("ℹ️  Motors: DISABLED (--no-motors)")
    elif not MOTORS_AVAILABLE:
        print("ℹ️  Motors: DISABLED (gpiozero/lgpio not available)")
    else:
        try:
            motors = MotorController(
                # m1_forward=args.m1_fwd, m1_backward=args.m1_bwd, m1_enable=args.m1_en,
                # m2_forward=args.m2_fwd, m2_backward=args.m2_bwd, m2_enable=args.m2_en,
            )
            print(f"✓  Motors: ENABLED")
        except Exception as e:
            print(f"⚠️  Motors: DISABLED (init failed: {e})")

    # ── Sanity check ──────────────────────────────────────────
    if robot_arm is None and motors is None:
        print("\n⚠️  WARNING: Neither arm nor motors are connected.")
        print("   The server will run but all incoming commands will be ignored.")

    # ── Start TCP server ──────────────────────────────────────
    print("=" * 55)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.tcp_port))
    server_sock.listen(1)
    print(f"✓ Listening on {args.host}:{args.tcp_port} …")
    print("  Waiting for client …\n")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"[server] Client connected from {addr}")
            handle_client(conn, robot_arm, motors, args.verbose)
            print("[server] Ready for next client …")
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
    finally:
        if motors is not None:
            motors.stop()
        server_sock.close()
        if robot_arm is not None:
            robot_arm.disconnect()
        print("[server] Clean exit.")


if __name__ == "__main__":
    main()