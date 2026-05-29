"""Ground-truth perception provider.

Reads object poses directly from `mjData` and computes visibility with
`mujoco.mj_ray` from the scene camera to each object center (occluded -> not
visible). Maintains a last-seen memory so the `SceneState` honors the partial
observability contract.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rtp.perception.api import ObjectState, Pose, SceneState


class GroundTruthPerception:
    """Implements `PerceptionProvider` using simulator state.

    Args:
        scene: the `MuJoCoScene` wrapper.
        scene_spec: the `SceneSpec` (for semantic labels: id/category/color).
        camera_name: camera used for the mj_ray occlusion test.
    """

    def __init__(self, scene, scene_spec, *, camera_name: str = "scene_cam") -> None:
        self.scene = scene
        self.scene_spec = scene_spec
        self.model = scene.model
        self.data = scene.data
        self.camera_name = camera_name
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self._memory: dict[str, ObjectState] = {}

    def _body_geom_ids(self, body_id: int) -> list[int]:
        start = self.model.body_geomadr[body_id]
        return list(range(start, start + self.model.body_geomnum[body_id]))

    def _aabb(self, body_id: int) -> np.ndarray:
        mins, maxs = [], []
        for gid in self._body_geom_ids(body_id):
            c = self.data.geom_xpos[gid]
            r = self.model.geom_rbound[gid]  # conservative sphere bound
            mins.append(c - r)
            maxs.append(c + r)
        if not mins:
            c = self.data.xpos[body_id]
            return np.array([c, c])
        return np.array([np.min(mins, axis=0), np.max(maxs, axis=0)])

    def _is_visible(self, body_id: int) -> bool:
        cam_pos = self.data.cam_xpos[self._cam_id].copy()
        target = self.data.xpos[body_id].copy()
        vec = target - cam_pos
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return True
        vec = vec / norm
        geomid = np.array([-1], dtype=np.int32)
        mujoco.mj_ray(self.model, self.data, cam_pos, vec, None, 1, -1, geomid)
        if geomid[0] < 0:
            return False
        return int(self.model.geom_bodyid[geomid[0]]) == body_id

    def observe(self, step: int) -> SceneState:
        visible: list[ObjectState] = []
        for obj in self.scene_spec.objects:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.id)
            if bid < 0:
                continue
            state = ObjectState(
                id=obj.id,
                name=obj.name,
                category=obj.category,
                color=obj.color,
                pose=Pose(position=self.data.xpos[bid].copy(),
                          quaternion=self.data.xquat[bid].copy()),
                aabb=self._aabb(bid),
                visible=self._is_visible(bid),
                confidence=1.0,
                last_seen_step=step,
            )
            if state.visible:
                self._memory[obj.id] = state
                visible.append(state)

        remembered = [s for oid, s in self._memory.items()
                      if oid not in {o.id for o in visible}]
        return SceneState(step=step, visible_objects=visible, remembered_objects=remembered)
