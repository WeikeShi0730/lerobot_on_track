# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# See the License for the specif

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30
ENABLE_RERUN = False
DEBUG_TELEOP = True
DEBUG_PRINT_INTERVAL_S = 1.0


def _format_debug_dict(values: dict, max_items: int | None = None) -> str:
    items = list(values.items())
    if max_items is not None:
        items = items[:max_items]
    return ", ".join(f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}" for key, value in items)


def _format_vector(values) -> str:
    return "[" + ", ".join(f"{float(value):+.3f}" for value in values) + "]"


def main():
    # Initialize the robot and teleoperator
    robot_config = SO100FollowerConfig(
        port="/dev/ttyACM0", id="so101_follower", use_degrees=True
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)  # or PhoneOS.ANDROID

    # Initialize the robot and teleoperator
    robot = SO100Follower(robot_config)
    teleop_device = Phone(teleop_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline to convert phone action to ee pose action to joint action
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    robot.connect()
    teleop_device.connect()

    # Rerun can crash on some Raspberry Pi/native library setups; keep phone teleop headless by default.
    if ENABLE_RERUN:
        init_rerun(session_name="phone_so100_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop. Move your phone to teleoperate the robot...")
    print("Debug: hold B1 while moving. If enabled=False, the phone is not commanding motion.")
    last_debug_print_s = 0.0
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = robot.get_observation()

        # Get teleop action
        phone_obs = teleop_device.get_action()

        # Phone -> EE pose -> Joints transition
        joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

        # Send action to robot
        sent_action = robot.send_action(joint_action)

        if DEBUG_TELEOP and t0 - last_debug_print_s >= DEBUG_PRINT_INTERVAL_S:
            raw_inputs = phone_obs.get("phone.raw_inputs", {})
            enabled = phone_obs.get("phone.enabled", False)
            phone_pos = phone_obs.get("phone.pos")
            motor_obs = {key: value for key, value in robot_obs.items() if key.endswith(".pos")}

            print("\n[teleop debug]")
            print(f"  phone.enabled={enabled} raw_inputs={{ {_format_debug_dict(raw_inputs)} }}")
            print(f"  phone.pos={_format_vector(phone_pos) if phone_pos is not None else None}")
            print(f"  robot_obs={{ {_format_debug_dict(motor_obs)} }}")
            print(f"  joint_action={{ {_format_debug_dict(joint_action)} }}")
            print(f"  sent_action={{ {_format_debug_dict(sent_action)} }}")
            last_debug_print_s = t0

        # Visualize
        if ENABLE_RERUN:
            log_rerun_data(observation=phone_obs, action=joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
