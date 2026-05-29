"""Replay a logged HDF5 episode in the viewer.

Usage:
    python scripts/replay_episode.py data/episodes/episodes.hdf5 --demo 0
"""

from __future__ import annotations

import sys

from _env import setup_env

setup_env()


def main(argv: list[str]) -> int:
    # TODO: open the HDF5 file, reconstruct the scene, and step through the
    #       recorded tool calls / states in the viewer.
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
