r"""Drive the Sockbot base in Isaac Sim with skid-steer wheel velocities.

Run this file with Isaac Sim's Python, not the repo's normal Python.
Example:
    C:\path\to\isaac-sim\python.bat examples\isaac_sim\drive_sockbot_base.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_USD_RELATIVE = Path("isaac_assets") / "assemblies" / "sockbot.usd"

WHEEL_JOINTS = (
    "wheel_front_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_left_joint",
    "wheel_rear_right_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd",
        type=Path,
        default=None,
        help="Sockbot assembly USD path. Defaults to .\\isaac_assets\\assemblies\\sockbot.usd.",
    )
    parser.add_argument(
        "--articulation-path",
        default="/World/sockbot/sockbot_base/Geometry/base_link",
        help="Prim path with the base articulation root.",
    )
    parser.add_argument("--linear", type=float, default=0.15, help="Forward velocity in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw velocity in rad/s. Positive turns left.")
    parser.add_argument("--duration", type=float, default=8.0, help="Drive duration in seconds.")
    parser.add_argument("--wheel-radius", type=float, default=0.038, help="Wheel radius in meters.")
    parser.add_argument("--track-width", type=float, default=0.20, help="Left/right wheel separation in meters.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the Isaac Sim UI.")
    parser.add_argument(
        "--flip-left",
        action="store_true",
        help="Flip left wheel sign if the base turns when it should drive straight.",
    )
    parser.add_argument(
        "--flip-right",
        action="store_true",
        help="Flip right wheel sign if the base turns when it should drive straight.",
    )
    return parser.parse_args()


def resolve_usd_path(usd_arg: Path | None) -> Path:
    if usd_arg is not None:
        return usd_arg.expanduser().resolve()

    cwd_default = (Path.cwd() / DEFAULT_USD_RELATIVE).resolve()
    if cwd_default.exists():
        return cwd_default

    script_default = (Path(__file__).resolve().parents[2] / DEFAULT_USD_RELATIVE).resolve()
    return script_default


def import_isaac_sim():
    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp

    return SimulationApp


def import_isaac_core():
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationActions
    except ImportError:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationActions

    return World, Articulation, add_reference_to_stage, ArticulationActions


def wheel_speeds(
    linear: float,
    yaw: float,
    wheel_radius: float,
    track_width: float,
    flip_left: bool,
    flip_right: bool,
) -> tuple[float, float]:
    left = (linear - yaw * track_width / 2.0) / wheel_radius
    right = (linear + yaw * track_width / 2.0) / wheel_radius

    if flip_left:
        left = -left
    if flip_right:
        right = -right

    return left, right


def joint_names_for(robot) -> list[str]:
    names = getattr(robot, "dof_names", None)
    if names is not None:
        return list(names)

    dof_names = getattr(robot, "get_dof_names", None)
    if dof_names is not None:
        return list(dof_names())

    raise RuntimeError("Could not read articulation DOF names from Isaac Sim.")


def dof_count_for(robot, joint_names: list[str]) -> int:
    count = getattr(robot, "num_dof", None)
    if count is not None:
        return int(count)

    count_fn = getattr(robot, "get_num_dof", None)
    if count_fn is not None:
        return int(count_fn())

    return len(joint_names)


def main() -> None:
    args = parse_args()
    usd_path = resolve_usd_path(args.usd)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD not found: {usd_path}")

    SimulationApp = import_isaac_sim()
    simulation_app = SimulationApp({"headless": args.headless})

    try:
        World, Articulation, add_reference_to_stage, ArticulationActions = import_isaac_core()

        world = World(stage_units_in_meters=1.0)
        add_reference_to_stage(str(usd_path), "/World/sockbot")

        robot = Articulation(args.articulation_path, name="sockbot_base")
        world.scene.add(robot)
        world.reset()

        joint_names = joint_names_for(robot)
        missing = [name for name in WHEEL_JOINTS if name not in joint_names]
        if missing:
            raise RuntimeError(
                "Wheel joints were not found on the articulation.\n"
                f"Missing: {missing}\n"
                f"Available DOFs: {joint_names}\n"
                f"Check --articulation-path. Current value: {args.articulation_path}"
            )

        wheel_indices = [joint_names.index(name) for name in WHEEL_JOINTS]
        dof_count = dof_count_for(robot, joint_names)

        left, right = wheel_speeds(
            args.linear,
            args.yaw,
            args.wheel_radius,
            args.track_width,
            args.flip_left,
            args.flip_right,
        )

        print(f"Loaded: {usd_path}")
        print(f"Articulation: {args.articulation_path}")
        print(f"Wheel joints: {dict(zip(WHEEL_JOINTS, wheel_indices, strict=True))}")
        print(f"Command: linear={args.linear:.3f} m/s yaw={args.yaw:.3f} rad/s")
        print(f"Wheel speeds: left={left:.3f} rad/s right={right:.3f} rad/s")

        elapsed = 0.0
        while simulation_app.is_running() and elapsed < args.duration:
            velocities = [0.0] * dof_count
            velocities[wheel_indices[0]] = left
            velocities[wheel_indices[1]] = right
            velocities[wheel_indices[2]] = left
            velocities[wheel_indices[3]] = right

            robot.apply_action(ArticulationActions(joint_velocities=[velocities]))
            world.step(render=not args.headless)
            elapsed += world.get_physics_dt()

        robot.apply_action(ArticulationActions(joint_velocities=[[0.0] * dof_count]))
        for _ in range(10):
            world.step(render=not args.headless)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
