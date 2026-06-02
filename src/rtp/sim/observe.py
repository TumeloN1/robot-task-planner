"""Utilities for observing an executing agent in MuJoCo.

The controllers advance the simulation internally, so observation is implemented
as a step observer registered on `MuJoCoScene`: every N physics steps, capture a
frame and/or sync a passive viewer.
"""

from __future__ import annotations

import time
from pathlib import Path


class VideoRecorder:
    """Record RGB frames from a `MuJoCoScene` to an MP4 while the scene steps."""

    def __init__(
        self,
        out_path: str | Path,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        camera: str = "scene_cam",
    ) -> None:
        self.out_path = Path(out_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = camera
        self._writer = None
        self._steps_per_frame: int | None = None
        self._step_count = 0
        self.frames = 0

    def attach(self, scene) -> None:
        """Attach to a scene and begin recording."""
        import imageio.v2 as imageio

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.out_path, fps=self.fps)
        self._steps_per_frame = max(1, round(1.0 / (self.fps * scene.model.opt.timestep)))
        scene.add_step_observer(self._on_step)
        self._write_frame(scene)

    def close(self, scene=None) -> None:
        """Detach and close the MP4 writer."""
        if scene is not None:
            try:
                scene.remove_step_observer(self._on_step)
            except ValueError:
                pass
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _on_step(self, scene) -> None:
        self._step_count += 1
        if self._steps_per_frame is None or self._step_count % self._steps_per_frame:
            return
        self._write_frame(scene)

    def _write_frame(self, scene) -> None:
        if self._writer is None:
            return
        frame = scene.render(width=self.width, height=self.height, camera=self.camera)
        self._writer.append_data(frame)
        self.frames += 1

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PassiveViewerSync:
    """Keep a MuJoCo passive viewer in sync with internal controller stepping."""

    def __init__(self, *, max_hz: float = 30.0, realtime: bool = True) -> None:
        self.max_hz = max_hz
        self.realtime = realtime
        self._viewer = None
        self._last_sync = 0.0
        self._start_wall: float | None = None
        self._start_sim: float | None = None

    def attach(self, scene):
        self._viewer = scene.launch_viewer()
        scene.add_step_observer(self._on_step)
        return self._viewer

    def close(self, scene=None) -> None:
        if scene is not None:
            try:
                scene.remove_step_observer(self._on_step)
            except ValueError:
                pass
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _on_step(self, scene) -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        now = time.monotonic()
        if self._start_wall is None:
            self._start_wall = now
            self._start_sim = float(scene.data.time)
        if self.realtime and self._start_sim is not None:
            sim_elapsed = float(scene.data.time) - self._start_sim
            wall_elapsed = now - self._start_wall
            sleep_for = sim_elapsed - wall_elapsed
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.05))
                now = time.monotonic()
        if now - self._last_sync < 1.0 / self.max_hz:
            return
        self._viewer.sync()
        self._last_sync = now
