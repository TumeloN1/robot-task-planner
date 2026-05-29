"""Inverse kinematics for the Panda end-effector (`grip_site`).

Primary path uses `mink` (QP-based, respects joint limits via a posture task and
solves only the arm DoFs - object free joints are naturally excluded since they
do not move the end-effector). A dependency-light damped-least-squares fallback
is provided for environments without a working QP backend. MVP grasps are
top-down, so a fixed downward orientation is the default target.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

try:
    import mink

    _HAS_MINK = True
except Exception:  # pragma: no cover - exercised only when mink is unavailable
    _HAS_MINK = False

# Fixed downward orientation (wxyz): 180 deg about x maps the site +z to world -z.
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
_QP_SOLVER = "daqp"


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


class TopDownIK:
    """Reusable IK solver bound to a scene's model (mink primary)."""

    def __init__(self, scene, *, position_cost: float = 1.0, orientation_cost: float = 1.0,
                 posture_cost: float = 1e-3) -> None:
        self.scene = scene
        self.model = scene.model
        self._arm_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS
        ])
        self._arm_dofadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in ARM_JOINTS
        ])
        self._use_mink = _HAS_MINK
        if self._use_mink:
            self._config = mink.Configuration(self.model)
            self._ee = mink.FrameTask(
                "grip_site", "site",
                position_cost=position_cost, orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            self._posture = mink.PostureTask(self.model, cost=posture_cost)

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray = TOP_DOWN_QUAT, *,
              q_init: np.ndarray | None = None, max_iters: int = 200, tol: float = 1e-3,
              dt: float = 0.01) -> np.ndarray | None:
        """Return the 7 arm joint targets reaching the pose, or None if infeasible."""
        q = (self.scene.data.qpos if q_init is None else q_init).copy()
        if self._use_mink:
            return self._solve_mink(q, target_pos, target_quat, max_iters, tol, dt)
        return self._solve_dls(q, target_pos, target_quat, max_iters=max_iters, tol=tol)

    def _solve_mink(self, q, target_pos, target_quat, max_iters, tol, dt):
        self._config.update(q)
        self._posture.set_target(q)
        rot = mink.SO3.from_matrix(_quat_to_mat(target_quat))
        self._ee.set_target(mink.SE3.from_rotation_and_translation(rot, np.asarray(target_pos)))
        tasks = [self._ee, self._posture]
        for _ in range(max_iters):
            vel = mink.solve_ik(self._config, tasks, dt, solver=_QP_SOLVER, damping=1e-3)
            self._config.integrate_inplace(vel, dt)
            if np.linalg.norm(self._ee.compute_error(self._config)) < tol:
                return self._config.q[self._arm_qadr].copy()
        # Accept if the position is close even if orientation lags slightly.
        if np.linalg.norm(self._ee.compute_error(self._config)[:3]) < 5e-3:
            return self._config.q[self._arm_qadr].copy()
        return None

    def _solve_dls(self, q, target_pos, target_quat, *, damping=1e-2, max_iters=300, tol=1e-3):
        """Damped-least-squares Jacobian IK on a cloned mjData (no QP solver)."""
        d = copy.deepcopy(self.scene.data)
        d.qpos[:] = q
        site = self.scene.grip_site_id
        tgt_mat = _quat_to_mat(target_quat)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        cols = self._arm_dofadr
        for _ in range(max_iters):
            mujoco.mj_forward(self.model, d)
            err_pos = target_pos - d.site_xpos[site]
            err_rot = np.zeros(3)
            cur_mat = d.site_xmat[site].reshape(3, 3)
            quat_err = np.zeros(4)
            mujoco.mju_mat2Quat(quat_err, (tgt_mat @ cur_mat.T).flatten())
            mujoco.mju_quat2Vel(err_rot, quat_err, 1.0)
            err = np.concatenate([err_pos, err_rot])
            if np.linalg.norm(err) < tol:
                return d.qpos[self._arm_qadr].copy()
            mujoco.mj_jacSite(self.model, d, jacp, jacr, site)
            jac = np.vstack([jacp, jacr])[:, cols]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            d.qpos[self._arm_qadr] += dq
        return None


def solve_ik(scene, target_pos: np.ndarray, target_quat: np.ndarray = TOP_DOWN_QUAT,
             **kwargs) -> np.ndarray | None:
    """Convenience wrapper; caches a `TopDownIK` solver on the scene."""
    solver = getattr(scene, "_ik_solver", None)
    if solver is None:
        solver = TopDownIK(scene)
        scene._ik_solver = solver
    return solver.solve(target_pos, target_quat, **kwargs)
