"""HDF5 episode recorder (robomimic-style).

Writes one group per episode (`data/demo_{i}`) so the dataset is directly
consumable by a future robomimic/BC pipeline and a learned failure model. Per
step we store the tool call, arguments (JSON), success, postcondition result,
and an explicit failure label.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py


class EpisodeRecorder:
    def __init__(self, out_path: str | Path) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.out_path, "a")
        self._data = self._file.require_group("data")
        self._episode_index = len(self._data.keys())
        self._buffer: list = []
        self._instruction = ""
        self._seed = 0

    def start_episode(self, instruction: str, seed: int) -> None:
        self._buffer = []
        self._instruction = instruction
        self._seed = seed

    def log_step(self, record) -> None:
        self._buffer.append(record)

    def end_episode(self, *, success: bool, num_replans: int,
                    injected_failures: list[str]) -> None:
        grp = self._data.create_group(f"demo_{self._episode_index}")
        self._episode_index += 1
        vstr = h5py.string_dtype()

        grp.create_dataset("tool", data=[r.tool for r in self._buffer], dtype=vstr)
        grp.create_dataset("args", data=[json.dumps(r.args) for r in self._buffer], dtype=vstr)
        grp.create_dataset("success", data=[int(r.success) for r in self._buffer])
        grp.create_dataset(
            "postcondition_pass",
            data=[(-1 if r.postcondition_pass is None else int(r.postcondition_pass))
                  for r in self._buffer],
        )
        grp.create_dataset(
            "failure_label",
            data=[(r.failure_label or "") for r in self._buffer], dtype=vstr,
        )

        grp.attrs["instruction"] = self._instruction
        grp.attrs["success"] = bool(success)
        grp.attrs["num_replans"] = int(num_replans)
        grp.attrs["injected_failures"] = json.dumps(injected_failures)
        grp.attrs["seed"] = int(self._seed)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
