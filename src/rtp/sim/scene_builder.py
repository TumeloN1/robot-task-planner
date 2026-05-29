"""Compose a full MuJoCo model from a `SceneSpec`.

Loads the vendored Franka Panda via `mujoco.MjSpec` (which resolves the model's
relative `meshdir`), adds an end-effector site, a floor, a table, a perception
camera, and one body per object. Using MjSpec avoids the `<include>` asset-path
pitfall and lets us edit the model programmatically (e.g. drop the Panda's
`home` keyframe, which would otherwise clash once object free joints change nq).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from rtp.sim.objects import ObjectSpec, SceneSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets"

# End-effector tool-center-point site, expressed in the hand body frame.
GRIP_SITE_NAME = "grip_site"
GRIP_SITE_POS = (0.0, 0.0, 0.1)

# Perception / render camera.
CAMERA_NAME = "scene_cam"

COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.80, 0.12, 0.12, 1.0),
    "green": (0.12, 0.70, 0.20, 1.0),
    "blue": (0.12, 0.30, 0.80, 1.0),
    "yellow": (0.90, 0.80, 0.12, 1.0),
}


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _add_object(worldbody, obj: ObjectSpec) -> None:
    rgba = list(COLORS.get(obj.color, (0.5, 0.5, 0.5, 1.0)))
    body = worldbody.add_body(name=obj.id, pos=list(obj.position), quat=_yaw_quat(obj.yaw))

    if not obj.is_static:
        body.add_freejoint()

    if obj.category == "block":
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02],
            rgba=rgba, mass=0.05, friction=[1.0, 0.05, 0.001],
        )
    elif obj.category == "mug":
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.045, 0.0],
            rgba=rgba, mass=0.12, friction=[1.0, 0.05, 0.001],
        )
    elif obj.category == "tray":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.09, 0.005],
                      rgba=rgba, mass=0.5)
        wall = 0.005
        h = 0.02
        for pos in ([0, 0.085, h], [0, -0.085, h]):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, wall, h],
                          pos=pos, rgba=rgba)
        for pos in ([0.085, 0, h], [-0.085, 0, h]):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[wall, 0.09, h],
                          pos=pos, rgba=rgba)
    else:
        raise ValueError(f"Unknown object category: {obj.category!r}")


def assemble_spec(
    scene: SceneSpec, *, assets_dir: Path | str = DEFAULT_ASSETS_DIR
) -> mujoco.MjSpec:
    """Build an MjSpec for the given scene (Panda + table + objects)."""
    assets_dir = Path(assets_dir)
    panda_xml = assets_dir / "menagerie" / "franka_emika_panda" / "panda.xml"
    if not panda_xml.exists():
        raise FileNotFoundError(
            f"Panda model not found at {panda_xml}. Run scripts/fetch_assets.py first."
        )

    spec = mujoco.MjSpec.from_file(str(panda_xml))

    # End-effector site at the TCP between the fingers.
    spec.body("hand").add_site(name=GRIP_SITE_NAME, pos=list(GRIP_SITE_POS),
                               size=[0.005, 0.005, 0.005])

    wb = spec.worldbody
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.1],
                rgba=[0.8, 0.8, 0.8, 1.0])

    table = wb.add_body(name="table", pos=[0.5, 0.0, 0.2])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.35, 0.45, 0.2],
                   rgba=[0.55, 0.42, 0.30, 1.0])

    wb.add_camera(name=CAMERA_NAME, pos=[0.5, -0.9, 0.9],
                  xyaxes=[1, 0, 0, 0, 0.5, 0.8])

    for obj in scene.objects:
        _add_object(wb, obj)

    # Drop the Panda 'home' keyframe: its qpos length no longer matches nq once
    # object free joints are added. The arm home pose is restored by joint name
    # in MuJoCoScene.reset().
    for key in list(spec.keys):
        spec.delete(key)

    return spec


def default_scene_spec() -> SceneSpec:
    """A simple deterministic scene for smoke testing: a red mug and a tray."""
    return SceneSpec(
        seed=0,
        objects=[
            ObjectSpec(id="red_mug", category="mug", color="red",
                       position=(0.45, -0.12, 0.43)),
            ObjectSpec(id="tray", category="tray", color="blue",
                       position=(0.55, 0.18, 0.41)),
        ],
    )
