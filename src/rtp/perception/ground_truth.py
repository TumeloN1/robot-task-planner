"""Ground-truth perception provider.

Reads object poses directly from `mjData` and computes visibility with
`mujoco.mj_ray` from the scene camera to each object center (occluded -> not
visible). Maintains a last-seen memory so the `SceneState` matches the partial
observability contract.
"""

from __future__ import annotations

from rtp.perception.api import ObjectState, SceneState


class GroundTruthPerception:
    """Implements `PerceptionProvider` using simulator state.

    Args:
        scene: the `MuJoCoScene` wrapper.
        camera_name: camera used for the mj_ray occlusion test.
    """

    def __init__(self, scene, camera_name: str = "scene_cam") -> None:
        self.scene = scene
        self.camera_name = camera_name
        self._memory: dict[str, ObjectState] = {}

    def observe(self, step: int) -> SceneState:
        """Build a `SceneState` from current sim bodies + occlusion test.

        TODO:
          - Read each tracked object's body pose + AABB from mjData.
          - Cast mj_ray from the camera to the object center; mark occluded
            objects visible=False.
          - Update self._memory for visible objects (set last_seen_step=step).
          - Return SceneState(visible_objects=..., remembered_objects=...).
        """
        raise NotImplementedError
