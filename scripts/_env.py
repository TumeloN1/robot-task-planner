"""Shared environment setup for scripts.

Import this first in every script to set the MuJoCo GL backend appropriately
(glfw on macOS, egl on Linux/Colab) and load .env for the Gemini API key.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def setup_env() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "egl"
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=_ROOT / ".env")
    except ImportError:
        pass


def planner_kwargs() -> dict:
    """Build GeminiPlanner kwargs from configs/planner.yaml."""
    import yaml

    cfg = yaml.safe_load((_ROOT / "configs" / "planner.yaml").read_text())
    keys = ("model", "temperature", "max_parse_repairs", "request_timeout_ms")
    return {k: cfg[k] for k in keys if k in cfg}
