"""Hugging Face utility helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_hf_login(token: str | None) -> str:
    if not token:
        return "HF token not provided (public repos only)."
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        return "HF login success."
    except Exception as exc:  # pragma: no cover
        return f"HF login failed: {exc}"


def maybe_download_adapter(repo_id: str, local_dir: Path, token: str | None = None) -> str:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir / repo_id.replace("/", "__")),
            token=token,
            local_dir_use_symlinks=False,
        )
        return path
    except Exception:
        return ""
