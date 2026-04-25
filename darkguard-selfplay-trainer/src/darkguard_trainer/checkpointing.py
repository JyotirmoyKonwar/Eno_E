"""Checkpoint and frozen pool metadata management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CheckpointMeta:
    role: str
    name: str
    round_idx: int
    score: float
    path: str
    created_at: str


def save_checkpoint(base_dir: Path, role: str, round_idx: int, payload: dict[str, Any], score: float) -> CheckpointMeta:
    role_dir = base_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    name = f"{role}_r{round_idx:04d}"
    ckpt_path = role_dir / f"{name}.json"
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    meta = CheckpointMeta(
        role=role,
        name=name,
        round_idx=round_idx,
        score=score,
        path=str(ckpt_path),
        created_at=datetime.now(UTC).isoformat(),
    )
    (role_dir / f"{name}.meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta


def load_latest_checkpoint(base_dir: Path, role: str) -> dict[str, Any] | None:
    role_dir = base_dir / role
    if not role_dir.exists():
        return None
    candidates = sorted(role_dir.glob("*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def write_frozen_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
