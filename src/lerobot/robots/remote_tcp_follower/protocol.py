#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Length-prefixed JSON protocol shared by remote TCP robot clients and servers."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise if the peer closes the connection."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by remote")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    """Receive one length-prefixed UTF-8 JSON object."""
    header = recv_exact(sock, 4)
    length = struct.unpack("<I", header)[0]
    raw = recv_exact(sock, length)
    msg = json.loads(raw.decode("utf-8"))
    if not isinstance(msg, dict):
        raise ValueError(f"Expected JSON object, got {type(msg).__name__}")
    return msg


def send_msg(sock: socket.socket, obj: dict[str, Any]) -> None:
    """Send one length-prefixed UTF-8 JSON object."""
    raw = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("<I", len(raw)) + raw)
