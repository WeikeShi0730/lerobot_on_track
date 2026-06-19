#!/usr/bin/env python3
"""
Run policy inference with the robot connected directly to this machine.

This avoids the TCP observation/action path used by lerobot_inference_client.py
and is intended for lower-latency local control.
"""

from dataclasses import dataclass, field
import logging
import math
import sys
import time
from typing import Any

import numpy as np
import yaml
import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.scripts.lerobot_inference_client import (
    JOINT_KEYS,
    InferenceStats,
    PolicyRunner,
    show_images,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


@dataclass
class DirectInferenceConfig:
    policy_path: str
    dataset_repo_id: str
    robot: RobotConfig | None = None
    robot_id: str = ""
    arm_port: str = ""
    cameras: str = ""
    device: str = "cpu"
    fps: int = 30
    actions_per_chunk: int = 0
    temporal_ensemble_coeff: float = float("nan")
    rename_map: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    show_images: bool = False
    calibrate: bool = True
    duration_s: float = 0.0


def normalize_legacy_cli_args(args: list[str]) -> list[str]:
    """Accept the TCP inference client's CLI spelling for direct inference."""
    aliases = {
        "policy-path": "policy_path",
        "dataset-repo-id": "dataset_repo_id",
        "actions-per-chunk": "actions_per_chunk",
        "temporal-ensemble-coeff": "temporal_ensemble_coeff",
        "rename-map": "rename_map",
        "show-images": "show_images",
        "dry-run": "dry_run",
        "arm-port": "arm_port",
        "robot-id": "robot_id",
        "cameras": "cameras",
    }
    bool_flags = {"show_images", "dry_run"}
    normalized: list[str] = []

    for arg in args:
        # Some shells/editors turn a leading double hyphen into an em dash.
        if arg.startswith("—"):
            arg = "--" + arg[1:]

        if not arg.startswith("--"):
            normalized.append(arg)
            continue

        body = arg[2:]
        name, separator, value = body.partition("=")
        mapped_name = aliases.get(name, name.replace("-", "_"))
        if separator:
            normalized.append(f"--{mapped_name}={value}")
        elif mapped_name in bool_flags:
            normalized.append(f"--{mapped_name}=true")
        else:
            normalized.append(f"--{mapped_name}")

    return normalized


def make_legacy_robot_config(robot_id: str, arm_port: str, cameras: str) -> RobotConfig:
    if not robot_id:
        raise ValueError("Missing robot id. Pass --robot-id=so101_follower or use --robot.type=...")
    if not arm_port:
        raise ValueError("Missing arm port. Pass --arm-port=COM7 or use --robot.port=...")

    config_cls = RobotConfig.get_choice_class(robot_id)
    args = [f"--port={arm_port}", f"--id={robot_id}"]
    if cameras:
        camera_configs = yaml.safe_load(cameras)
        if "base" in camera_configs and "rotation" not in camera_configs["base"]:
            camera_configs["base"]["rotation"] = 180
        args.append(f"--cameras={camera_configs}")
    return draccus.parse(config_cls, args=args)


def robot_observation_to_policy_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert local robot observations to the policy format used by PolicyRunner."""
    obs: dict[str, Any] = {
        "observation.state": np.array([raw_obs.get(key, 0.0) for key in JOINT_KEYS], dtype=np.float32)
    }

    for key, value in raw_obs.items():
        if key in JOINT_KEYS:
            continue
        if key.startswith("observation.images."):
            obs[key] = value
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            obs[f"observation.images.{key}"] = value

    return obs


def run_direct_inference(
    robot: Robot,
    policy: PolicyRunner,
    fps: int,
    dry_run: bool,
    show_imgs: bool = False,
    duration_s: float | None = None,
) -> None:
    stats = InferenceStats(fps)
    control_dt = 1.0 / fps
    step = 0
    start_t = time.perf_counter()

    policy.reset()
    n_action_steps = policy.n_action_steps
    temporal_ensemble_coeff = policy.temporal_ensemble_coeff
    te_status = (
        f", temporal_ensemble_coeff={temporal_ensemble_coeff}"
        if temporal_ensemble_coeff is not None
        else ""
    )
    print(f"Running direct inference at {fps} Hz  (n_action_steps={n_action_steps}{te_status})")
    print("Press Ctrl+C to stop.\n")

    next_step_t = time.perf_counter()
    obs: dict[str, Any] | None = None

    try:
        while True:
            if duration_s is not None and time.perf_counter() - start_t >= duration_s:
                return

            if step % n_action_steps == 0:
                t0 = time.perf_counter()
                raw_obs = robot.get_observation()
                obs = robot_observation_to_policy_observation(raw_obs)
                stats.record_obs(time.perf_counter() - t0)
                if show_imgs:
                    show_images(obs)

            if obs is None:
                logging.warning("step=%d: waiting for first complete observation", step)
                precise_sleep(min(control_dt, 0.1))
                continue

            t0 = time.perf_counter()
            action_dict = policy.predict(obs)
            stats.record_infer(time.perf_counter() - t0)

            sleep_t = next_step_t - time.perf_counter()
            if sleep_t > 0:
                precise_sleep(sleep_t)
            next_step_t += control_dt

            stats.record_loop(time.perf_counter())

            if dry_run:
                vals = [f"{action_dict[k]:6.2f}" for k in JOINT_KEYS]
                print(f"\r  [dry-run] step={step:5d}  {vals}", end="", flush=True)
            else:
                t0 = time.perf_counter()
                robot.send_action(action_dict)
                stats.record_send(time.perf_counter() - t0)

            step += 1
            stats.maybe_print(step)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        stats.print_summary()


@parser.wrap()
def direct_inference(cfg: DirectInferenceConfig) -> None:
    init_logging()
    robot_cfg = cfg.robot or make_legacy_robot_config(cfg.robot_id, cfg.arm_port, cfg.cameras)
    actions_per_chunk = cfg.actions_per_chunk if cfg.actions_per_chunk > 0 else None
    temporal_ensemble_coeff = (
        None if math.isnan(cfg.temporal_ensemble_coeff) else cfg.temporal_ensemble_coeff
    )
    duration_s = cfg.duration_s if cfg.duration_s > 0 else None

    policy = PolicyRunner(
        cfg.policy_path,
        dataset_repo_id=cfg.dataset_repo_id,
        device=cfg.device,
        actions_per_chunk=actions_per_chunk,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        rename_map=cfg.rename_map,
    )

    robot = make_robot_from_config(robot_cfg)
    try:
        robot.connect(calibrate=cfg.calibrate)
        print("\n" + "=" * 45)
        print("  LEROBOT DIRECT INFERENCE")
        print("=" * 45)
        if cfg.dry_run:
            print("  DRY-RUN - actions will NOT be sent")
        print(f"  Robot:  {robot.name}")
        print(f"  FPS:    {cfg.fps}")
        print(f"  Device: {cfg.device}")
        print()

        run_direct_inference(
            robot,
            policy,
            fps=cfg.fps,
            dry_run=cfg.dry_run,
            show_imgs=cfg.show_images,
            duration_s=duration_s,
        )
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("Robot disconnected.")


def main() -> None:
    register_third_party_plugins()
    sys.argv[1:] = normalize_legacy_cli_args(sys.argv[1:])
    direct_inference()


if __name__ == "__main__":
    main()
