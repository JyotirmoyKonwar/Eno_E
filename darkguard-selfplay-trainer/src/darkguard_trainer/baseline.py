"""Baseline judge interfaces (including Eno_E stub)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaselineJudge(ABC):
    @abstractmethod
    def score_episode(self, episode_trace: dict[str, Any]) -> float:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class NoOpBaselineJudge(BaselineJudge):
    def score_episode(self, episode_trace: dict[str, Any]) -> float:
        _ = episode_trace
        return 0.0

    def name(self) -> str:
        return "noop"


class EnoEBaselineJudge(BaselineJudge):
    """Conservative baseline placeholder. Replace with real Eno_E calls later."""

    def score_episode(self, episode_trace: dict[str, Any]) -> float:
        # TODO: integrate actual Eno_E inference endpoint / local runner.
        # This must remain evaluation-only and never leak hidden labels to policies.
        outcome = str(episode_trace.get("outcome_summary", ""))
        reward = float(episode_trace.get("cumulative_reward", 0.0))
        return 0.5 if "safe" in outcome else max(-0.2, min(0.3, reward * 0.1))

    def name(self) -> str:
        return "eno_e_stub"
