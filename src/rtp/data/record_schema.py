"""Episode record schema (robomimic-style HDF5 layout).

One group per episode (`data/demo_{i}`) with per-step datasets and episode
attrs. The explicit `failure_label` per step is what makes the logs reusable
for a future learned failure model and behavior cloning.

Per-step fields:
  obs/object_states : serialized SceneState (JSON string)
  obs/proprio       : robot joint state (float array)
  tool_call         : tool name (string)
  args              : JSON-serialized arguments (string)
  primitive_result  : success flag (bool/int)
  postcondition_pass: check outcome (int; -1 if not a check step)
  failure_label     : FailureType value or "" (string)

Episode attrs:
  instruction, success, num_replans, injected_failures, seed
"""

from __future__ import annotations

PER_STEP_FIELDS = (
    "obs/object_states",
    "obs/proprio",
    "tool_call",
    "args",
    "primitive_result",
    "postcondition_pass",
    "failure_label",
)

EPISODE_ATTRS = (
    "instruction",
    "success",
    "num_replans",
    "injected_failures",
    "seed",
)
