"""Replay/archive utilities for hard episode retention."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EpisodeArchive:
    max_size: int
    _items: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.max_size)

    def add(self, episode: dict[str, Any]) -> None:
        self._items.append(episode)

    def sample_recent(self, n: int = 5) -> list[dict[str, Any]]:
        return list(self._items)[-n:]

    def __len__(self) -> int:
        return len(self._items)
