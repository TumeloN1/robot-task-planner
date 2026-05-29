"""Vendor the Franka Emika Panda MJCF from MuJoCo Menagerie.

Downloads only the `franka_emika_panda/` subtree into
`assets/menagerie/franka_emika_panda/` (gitignored). Run once after setup.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ZIP = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
SUBDIR = "franka_emika_panda/"
DEST = Path(__file__).resolve().parent.parent / "assets" / "menagerie" / "franka_emika_panda"


def main() -> int:
    if DEST.exists() and any(DEST.iterdir()):
        print(f"Panda model already present at {DEST}")
        return 0

    print("Downloading MuJoCo Menagerie (this fetches the whole repo zip once)...")
    with urllib.request.urlopen(REPO_ZIP) as resp:
        data = resp.read()

    DEST.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            # paths look like: mujoco_menagerie-main/franka_emika_panda/...
            parts = member.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if not rel.startswith(SUBDIR) or member.endswith("/"):
                continue
            out = DEST / rel[len(SUBDIR):]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))
            extracted += 1

    print(f"Extracted {extracted} files to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
