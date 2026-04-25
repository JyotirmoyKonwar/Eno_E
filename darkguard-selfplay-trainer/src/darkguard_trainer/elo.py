"""ELO helpers for self-play league ranking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EloRatings:
    consumer: float = 1200.0
    designer: float = 1200.0
    baseline: float = 1200.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k_factor: float) -> tuple[float, float]:
    exp_a = expected_score(rating_a, rating_b)
    exp_b = expected_score(rating_b, rating_a)
    new_a = rating_a + k_factor * (score_a - exp_a)
    new_b = rating_b + k_factor * ((1.0 - score_a) - exp_b)
    return round(new_a, 4), round(new_b, 4)
