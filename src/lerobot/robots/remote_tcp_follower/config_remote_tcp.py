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

from dataclasses import dataclass, field

from ..config import RobotConfig


def default_so101_joint_keys() -> list[str]:
    return [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]


@RobotConfig.register_subclass("remote_tcp_follower")
@dataclass
class RemoteTcpFollowerConfig(RobotConfig):
    """Config for a robot controlled through the existing JSON-over-TCP protocol."""

    host: str
    tcp_port: int = 2222
    timeout_s: float = 10.0
    joint_keys: list[str] = field(default_factory=default_so101_joint_keys)
    camera_names: list[str] = field(default_factory=lambda: ["front"])
    image_shape: tuple[int, int, int] = (480, 640, 3)
    request_arm_mode: bool = True
    request_images: bool = True
