"""Shared environment setup for scripts.

Import this first in every script to set the MuJoCo GL backend appropriately
(glfw on macOS, egl on Linux/Colab) and load .env for the Gemini API key.
"""

from __future__ import annotations

import os
import platform


def setup_env() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "egl"
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
