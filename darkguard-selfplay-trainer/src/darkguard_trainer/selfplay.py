"""Self-play scheduler and opponent sampling logic."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class OpponentEntry:
    name: str
    elo: float
    role: str
    checkpoint: str = ""


@dataclass(slots=True)
class LeaguePools:
    consumer_pool: list[OpponentEntry] = field(default_factory=list)
    designer_pool: list[OpponentEntry] = field(default_factory=list)

    def add(self, entry: OpponentEntry) -> None:
        pool = self.consumer_pool if entry.role == "consumer" else self.designer_pool
        pool.append(entry)


def phase_for_round(round_idx: int) -> str:
    mod = round_idx % 4
    if mod == 1:
        return "train_consumer"
    if mod == 2:
        return "train_designer"
    if mod == 3:
        return "evaluation"
    return "snapshot"


def sample_opponent(active_elo: float, pool: list[OpponentEntry], rng: random.Random) -> OpponentEntry | None:
    if not pool:
        return None
    near = sorted(pool, key=lambda p: abs(p.elo - active_elo))
    choices: list[OpponentEntry] = []
    choices.extend(near[: max(1, min(3, len(near)))])
    if len(near) > 3:
        choices.append(near[-1])  # stronger occasionally
    choices.append(rng.choice(pool))  # random historical
    return rng.choice(choices)


def recent_opponents(pool: list[OpponentEntry], window: int) -> list[OpponentEntry]:
    if not pool:
        return []
    if window <= 0:
        return pool
    return pool[-window:]
