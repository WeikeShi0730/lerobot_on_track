#!/usr/bin/env python3
"""
lerobot_gamepad_control_server.py  –  runs on server Raspberry Pi
================================================
Receives action dicts from the Windows gamepad client (or policy client)
over TCP and drives the SO-101 follower arm.

Usage:
    lerobot-gamepad-control-server
    lerobot-gamepad-control-server --port /dev/ttyACM0 --robot-id so101_follower
    lerobot-gamepad-control-server --host 0.0.0.0 --tcp-port 5555

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

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


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


# Initialize robot
def connect_robot(port: str, robot_id: str) -> SO101Follower:
    print(f"✓ Connecting to SO-101 on {port} (id={robot_id}) …")
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)
    robot.connect()
    print("✓ Robot connected.")
    return robot


def get_observation_dict(robot: SO101Follower) -> dict:
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

def handle_client(conn: socket.socket, robot: SO101Follower, verbose: bool) -> None:
    stats = ServerStats()
    try:
        while True:
            msg = recv_msg(conn)
            mtype = msg.get("type")

            # ── action ──────────────────────────────────────────────────
            if mtype == "action":
                action_dict: dict = msg["action"]   # {joint_key: float, …}
                recv_t = time.perf_counter()        # arrival time, before exec
                t0 = time.perf_counter()
                robot.send_action(action_dict)
                exec_s = time.perf_counter() - t0
                stats.record_action(recv_t, exec_s)
                stats.maybe_print()

                if verbose:
                    vals = [f"{action_dict.get(k, 0):.1f}" for k in JOINT_KEYS]
                    print(f"[server] action exec={exec_s*1000:.2f}ms  pos={vals}")

            # ── ping → pong ─────────────────────────────────────────────
            elif mtype == "ping":
                stats.record_ping()
                send_msg(conn, {"type": "pong", "seq": msg.get("seq")})

            # ── observation request ─────────────────────────────────────
            elif mtype == "obs_request":
                obs = get_observation_dict(robot)
                send_msg(conn, {"type": "obs", "data": obs})

            # ── graceful disconnect ─────────────────────────────────────
            elif mtype == "disconnect":
                print("[server] Client requested disconnect.")
                break

            else:
                print(f"[server] Unknown message type: {mtype!r}")

    except ConnectionError as e:
        print(f"[server] Connection lost: {e}")
    finally:
        stats.print_summary()
        conn.close()
        print("[server] Client connection closed.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SO-101 robot TCP server (Raspberry Pi)")
    parser.add_argument("--port",      default="/dev/ttyACM0", help="Serial port for SO-101")
    parser.add_argument("--robot-id",  default="so101_follower")
    parser.add_argument("--host",      default="0.0.0.0",      help="Bind address")
    parser.add_argument("--tcp-port",  type=int, default=2222,  help="TCP port")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    robot = connect_robot(args.port, args.robot_id)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.tcp_port))
    server_sock.listen(1)
    print(f"✓ Listening on {args.host}:{args.tcp_port} …")
    print("  Waiting for Windows client …\n")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"[server] Client connected from {addr}")
            handle_client(conn, robot, args.verbose)
            print("[server] Ready for next client …")
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
    finally:
        server_sock.close()
        robot.disconnect()
        print("[server] Clean exit.")


if __name__ == "__main__":
    main()