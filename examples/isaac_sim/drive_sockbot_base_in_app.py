r"""Drive Sockbot from a running Isaac Sim app via the VS Code extension.

Use this with Isaac Sim already open:
1. In Isaac Sim, enable the `isaacsim.code_editor.vscode` extension.
2. Open this file in VS Code.
3. Click the Isaac Sim logo in the VS Code Activity Bar.
4. Click `Run`.

This script is not a standalone launcher. It is meant to execute inside
Isaac Sim's existing Python process.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, UsdGeom, UsdLux


KNOWN_REPO_ROOT = Path(r"C:\Users\vicsh\Projects\lerobot_on_track")
BASE_USD_RELATIVE_PATH = Path("isaac_assets") / "sockbot_base" / "sockbot_base.usda"
ARM_USD_RELATIVE_PATH = Path("isaac_assets") / "so101_new_calib" / "so101_new_calib.usda"


def resolve_repo_path(relative_path: Path) -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / relative_path,
        KNOWN_REPO_ROOT / relative_path,
        Path.cwd() / relative_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {relative_path}. Checked:\n{checked}")


BASE_USD_PATH = resolve_repo_path(BASE_USD_RELATIVE_PATH)
ARM_USD_PATH = resolve_repo_path(ARM_USD_RELATIVE_PATH)
BASE_REFERENCE_PATH = "/World/sockbot_base"
ARM_REFERENCE_PATH = f"{BASE_REFERENCE_PATH}/so101_arm"
GROUND_PATH = "/World/GroundPlane"
SUN_PATH = "/World/Sun"
CAMERA_PATH = "/World/Camera"
REFERENCE_MARKERS_PATH = "/World/ReferenceMarkers"

LINEAR_M_S = 0.2
YAW_RAD_S = 0.0
WHEEL_RADIUS_M = 0.038
TRACK_WIDTH_M = 0.20
DRIVE_DAMPING = 100.0
DRIVE_MAX_FORCE = 1000.0
USE_KINEMATIC_DEBUG_MOTION = True

# Change one of these to True if a straight command spins in place.
FLIP_LEFT = False
FLIP_RIGHT = False

WHEEL_JOINT_PATHS = {
    "wheel_front_left_joint": f"{BASE_REFERENCE_PATH}/Physics/wheel_front_left_joint",
    "wheel_front_right_joint": f"{BASE_REFERENCE_PATH}/Physics/wheel_front_right_joint",
    "wheel_rear_left_joint": f"{BASE_REFERENCE_PATH}/Physics/wheel_rear_left_joint",
    "wheel_rear_right_joint": f"{BASE_REFERENCE_PATH}/Physics/wheel_rear_right_joint",
}


def print_prim_tree(prim, indent: int = 0, max_depth: int = 4) -> None:
    print("  " * indent + str(prim.GetPath()))
    if indent >= max_depth:
        return
    for child in prim.GetChildren():
        print_prim_tree(child, indent + 1, max_depth)


def wheel_speeds() -> tuple[float, float]:
    left = (LINEAR_M_S - YAW_RAD_S * TRACK_WIDTH_M / 2.0) / WHEEL_RADIUS_M
    right = (LINEAR_M_S + YAW_RAD_S * TRACK_WIDTH_M / 2.0) / WHEEL_RADIUS_M

    if FLIP_LEFT:
        left = -left
    if FLIP_RIGHT:
        right = -right

    return left, right


def set_float_attr(prim, name: str, value: float) -> None:
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
    attr.Set(value)


def set_wheel_drive_targets(stage, left_speed: float, right_speed: float) -> None:
    speeds_by_joint = {
        "wheel_front_left_joint": left_speed,
        "wheel_front_right_joint": right_speed,
        "wheel_rear_left_joint": left_speed,
        "wheel_rear_right_joint": right_speed,
    }

    for joint_name, speed in speeds_by_joint.items():
        prim = stage.GetPrimAtPath(WHEEL_JOINT_PATHS[joint_name])
        if not prim:
            print(f"Expected wheel joint prim not found: {WHEEL_JOINT_PATHS[joint_name]}")
            print("Current /World prim tree:")
            print_prim_tree(stage.GetPrimAtPath("/World"))
            raise RuntimeError("Wheel joint path is missing; see printed prim tree above.")

        set_float_attr(prim, "drive:angular:physics:targetVelocity", speed)
        set_float_attr(prim, "drive:angular:physics:damping", DRIVE_DAMPING)
        set_float_attr(prim, "drive:angular:physics:maxForce", DRIVE_MAX_FORCE)


def get_translate_op(prim):
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xform.AddTranslateOp()


def set_xform(prim, translate: tuple[float, float, float], rotate_xyz: tuple[float, float, float] = (0, 0, 0)) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))


def add_viewing_scene(stage) -> None:
    if not stage.GetPrimAtPath(GROUND_PATH):
        GroundPlane(prim_path=GROUND_PATH, z_position=-0.038, size=4.0, color=np.array([0.18, 0.18, 0.18]))

    if stage.GetPrimAtPath(REFERENCE_MARKERS_PATH):
        stage.RemovePrim(REFERENCE_MARKERS_PATH)
    markers_root = UsdGeom.Xform.Define(stage, REFERENCE_MARKERS_PATH)
    markers_root.GetPrim()
    for idx, x in enumerate(np.linspace(-1.0, 1.0, 9)):
        marker = UsdGeom.Cube.Define(stage, f"{REFERENCE_MARKERS_PATH}/x_marker_{idx}")
        marker.CreateSizeAttr(1.0)
        set_xform(marker.GetPrim(), translate=(float(x), -0.35, -0.034))
        marker.GetDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.85, 0.12)])
        marker.AddScaleOp().Set(Gf.Vec3f(0.012, 0.08, 0.004))

    origin = UsdGeom.Cube.Define(stage, f"{REFERENCE_MARKERS_PATH}/origin_marker")
    origin.CreateSizeAttr(1.0)
    set_xform(origin.GetPrim(), translate=(0.0, 0.0, -0.032))
    origin.GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.65, 1.0)])
    origin.AddScaleOp().Set(Gf.Vec3f(0.08, 0.08, 0.006))

    if not stage.GetPrimAtPath(SUN_PATH):
        sun = UsdLux.DistantLight.Define(stage, SUN_PATH)
        sun.CreateIntensityAttr(500.0)
        sun.CreateAngleAttr(0.6)
        set_xform(sun.GetPrim(), translate=(0, 0, 1), rotate_xyz=(-45, 0, 35))

    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    set_xform(camera.GetPrim(), translate=(0.55, -0.55, 0.34), rotate_xyz=(62, 0, 42))
    camera.CreateFocalLengthAttr(28.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def install_kinematic_debug_motion(stage, timeline, linear_m_s: float) -> None:
    base_prim = stage.GetPrimAtPath(BASE_REFERENCE_PATH)
    translate_op = get_translate_op(base_prim)
    start = translate_op.Get() or Gf.Vec3d(0, 0, 0)
    state = {"x": float(start[0]), "last_time": None, "sub": None}

    def on_update(event) -> None:
        current_time = timeline.get_current_time()
        if state["last_time"] is None:
            state["last_time"] = current_time
            return

        dt = max(0.0, current_time - state["last_time"])
        state["last_time"] = current_time
        state["x"] += linear_m_s * dt
        translate_op.Set(Gf.Vec3d(state["x"], float(start[1]), float(start[2])))

    app = omni.kit.app.get_app()
    state["sub"] = app.get_update_event_stream().create_subscription_to_pop(on_update, name="sockbot_debug_motion")
    globals()["_sockbot_debug_motion_subscription"] = state["sub"]


stage = omni.usd.get_context().get_stage()
for prim in list(stage.GetPrimAtPath("/World").GetChildren()):
    if prim.GetName().startswith(("sockbot", "so101")):
        stage.RemovePrim(prim.GetPath())

add_reference_to_stage(str(BASE_USD_PATH), BASE_REFERENCE_PATH)
stage.Load(BASE_REFERENCE_PATH)
add_reference_to_stage(str(ARM_USD_PATH), ARM_REFERENCE_PATH)
stage.Load(ARM_REFERENCE_PATH)
set_xform(stage.GetPrimAtPath(ARM_REFERENCE_PATH), translate=(0.06, 0, 0.03))

left_speed, right_speed = wheel_speeds()
set_wheel_drive_targets(stage, left_speed, right_speed)
add_viewing_scene(stage)

print(f"Loaded base: {BASE_USD_PATH}")
print(f"Loaded arm: {ARM_USD_PATH}")
print(f"Wheel speeds: left={left_speed:.3f} rad/s right={right_speed:.3f} rad/s")
print("Wheel drive targets are set. Press Stop in Isaac Sim to stop the simulation.")

timeline = omni.timeline.get_timeline_interface()
if USE_KINEMATIC_DEBUG_MOTION:
    install_kinematic_debug_motion(stage, timeline, LINEAR_M_S)
    print("Kinematic debug motion is enabled, so the base xform will move visibly.")
timeline.stop()
timeline.play()
