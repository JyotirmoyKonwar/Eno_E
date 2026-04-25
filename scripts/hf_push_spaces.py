#!/usr/bin/env python3
"""Local helper: deploy both DarkGuard Spaces (reads HF_TOKEN from .env)."""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_hf_space_folder.py"


def _load_hf_token() -> str:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        print("Missing .env with HF_TOKEN", file=sys.stderr)
        sys.exit(1)
    for line in env_path.read_text().splitlines():
        m = re.match(r"^\s*HF_TOKEN\s*=\s*(.+)$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    print("HF_TOKEN not set in .env", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    token = _load_hf_token()
    env = {**os.environ, "HF_TOKEN": token}
    pairs = [
        ("Jyo-K/darkguard-openenv", "darkguard-openenv"),
        ("Jyo-K/darkguard-selfplay-trainer", "darkguard-selfplay-trainer"),
    ]
    for repo_id, folder in pairs:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-id",
                repo_id,
                "--folder",
                folder,
                "--root",
                str(ROOT),
                "--message",
                f"Local sync: {folder}",
            ],
            env=env,
            check=True,
            cwd=str(ROOT),
        )
    print("OK: both Spaces updated.", flush=True)


if __name__ == "__main__":
    main()
