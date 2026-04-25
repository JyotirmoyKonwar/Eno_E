"""Utility helpers for IDs and lightweight text processing."""

from __future__ import annotations

import uuid


def new_episode_id() -> str:
    return f"dg-{uuid.uuid4().hex[:12]}"


def norm_text(value: str | None) -> str:
    return (value or "").strip().lower()
