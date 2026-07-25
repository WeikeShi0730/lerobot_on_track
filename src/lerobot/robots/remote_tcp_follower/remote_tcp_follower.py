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

from __future__ import annotations

import base64
import logging
import socket
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_remote_tcp import RemoteTcpFollowerConfig
from .protocol import recv_msg, send_msg

logger = logging.getLogger(__name__)


def _decode_image(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded image into an RGB uint8 array."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to decode remote TCP camera images") from exc

    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Remote TCP server returned an image that OpenCV could not decode")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class RemoteTcpFollower(Robot):
    """LeRobot ``Robot`` adapter for the existing remote inference TCP server."""

    config_class = RemoteTcpFollowerConfig
    name = "remote_tcp_follower"

    def __init__(self, config: RemoteTcpFollowerConfig):
        super().__init__(config)
        self.config = config
        self._sock: socket.socket | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        motor_features = {key: float for key in self.config.joint_keys}
        camera_features = {name: self.config.image_shape for name in self.config.camera_names}
        return {**motor_features, **camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self.config.joint_keys}

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.config.timeout_s)
        try:
            sock.connect((self.config.host, self.config.tcp_port))
            if self.config.request_arm_mode:
                send_msg(sock, {"type": "mode_request", "mode": "arm"})
                response = recv_msg(sock)
                if response.get("type") == "mode_response" and not response.get("ok", False):
                    logger.warning("Remote TCP arm mode denied: %s", response.get("reason", "unknown"))
            self._sock = sock
        except Exception:
            sock.close()
            raise

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        sock = self._require_socket()
        send_msg(sock, {"type": "full_obs_request", "images": self.config.request_images})
        msg = recv_msg(sock)
        if msg.get("type") != "full_obs":
            raise RuntimeError(f"Unexpected remote TCP response: {msg.get('type')}")

        obs: dict[str, Any] = {}
        state = msg.get("state", [])
        if len(state) != len(self.config.joint_keys):
            raise ValueError(
                f"Remote TCP server returned {len(state)} joint values, expected {len(self.config.joint_keys)}"
            )
        obs.update({key: float(value) for key, value in zip(self.config.joint_keys, state, strict=True)})

        for camera_name, b64_img in msg.get("images", {}).items():
            if camera_name in self.config.camera_names:
                obs[camera_name] = _decode_image(b64_img)

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sock = self._require_socket()
        action_dict = {key: float(value) for key, value in action.items() if key in self.action_features}
        send_msg(sock, {"type": "action", "action": action_dict})
        return action_dict

    @check_if_not_connected
    def disconnect(self) -> None:
        sock = self._require_socket()
        try:
            send_msg(sock, {"type": "disconnect"})
        except OSError:
            logger.debug("Remote TCP disconnect message failed", exc_info=True)
        finally:
            sock.close()
            self._sock = None

    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("Remote TCP follower is not connected")
        return self._sock
