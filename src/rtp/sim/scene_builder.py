"""Compose a full MJCF model from a `SceneSpec`.

Starts from `assets/scene_base.xml` (table, lights, camera, and an <include> of
the vendored Panda model) and injects one body per object from
`assets/objects/*.xml`, positioned per the spec.
"""

from __future__ import annotations

from rtp.sim.objects import SceneSpec


def build_model_xml(spec: SceneSpec, *, base_xml_path: str, objects_dir: str) -> str:
    """Return an assembled MJCF XML string for the given scene spec.

    TODO:
      - Load scene_base.xml.
      - For each ObjectSpec, instantiate the matching object snippet with a
        freejoint, position, yaw, and color material.
      - Return the merged XML (string templating or dm_control PyMJCF).
    """
    raise NotImplementedError
