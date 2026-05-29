"""MuJoCo scene wrapper.

Owns the `mjModel` / `mjData` pair and exposes a small, stable surface
(step / reset / render / clone_data) used by the rest of the system. Keeps the
model/data separation that MuJoCo encourages so feasibility checks can run on a
cloned `mjData` without touching the live simulation.
"""

from __future__ import annotations


class MuJoCoScene:
    """Loads a composed MJCF and drives stepping/rendering.

    Args:
        model_xml: path to (or string of) the assembled MJCF.
        timestep: physics timestep (s).
    """

    def __init__(self, model_xml: str, timestep: float = 0.002) -> None:
        self.model_xml = model_xml
        self.timestep = timestep
        # TODO: mujoco.MjModel.from_xml_path/string, mujoco.MjData(model),
        #       resolve the "home" keyframe, cache body/site/actuator ids.

    def reset(self) -> None:
        """Reset data to the home keyframe."""
        raise NotImplementedError

    def step(self, n: int = 1) -> None:
        """Advance the physics `n` steps."""
        raise NotImplementedError

    def clone_data(self):
        """Return a deep copy of mjData for off-line feasibility checks."""
        raise NotImplementedError

    def render(self):
        """Return an RGB frame (offscreen) or drive the interactive viewer."""
        raise NotImplementedError
