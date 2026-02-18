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
    action_count = 0
    t0 = time.time()
    try:
        while True:
            msg = recv_msg(conn)
            mtype = msg.get("type")

            # ── action ──────────────────────────────────────────────────
            if mtype == "action":
                action_dict: dict = msg["action"]   # {joint_key: float, …}
                robot.send_action(action_dict)
                action_count += 1

                if verbose and action_count % 90 == 0:
                    hz = action_count / max(time.time() - t0, 1e-6)
                    vals = [f"{action_dict.get(k, 0):.1f}" for k in JOINT_KEYS]
                    print(f"[server] {action_count} actions  (~{hz:.1f} Hz)  pos={vals}")

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