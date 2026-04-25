#!/usr/bin/env python3
"""Upload a single project folder to a Hugging Face Space (used by CI and locally)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Patterns relative to the uploaded folder; see huggingface_hub pathspec rules.
IGNORE_PATTERNS = [
    ".venv",
    ".venv/**",
    "**/.venv/**",
    "__pycache__",
    "**/__pycache__/**",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    "*.pyc",
    "**/*.pyc",
    ".mypy_cache/**",
    "outputs/**",
    ".git/**",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. Jyo-K/darkguard-openenv")
    parser.add_argument("--folder", required=True, help="Path relative to repo root")
    parser.add_argument("--message", default="Deploy from CI", help="Commit message")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root (default: cwd)",
    )
    args = parser.parse_args()
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root).resolve()
    folder = (root / args.folder).resolve()
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"Uploading {folder} → {args.repo_id} (repo_type=space)", flush=True)
    api.upload_folder(
        folder_path=str(folder),
        path_in_repo=".",
        repo_id=args.repo_id,
        repo_type="space",
        commit_message=args.message,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
