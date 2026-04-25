"""Tournament and holdout evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable

from .baseline import BaselineJudge
from .env_client import RemoteEnvClient
from .model_utils import PolicyModel
from .rollout import run_consumer_episode


@dataclass(slots=True)
class EvalSummary:
    mean_reward: float
    safe_rate: float
    invalid_rate: float
    baseline_score: float
    episodes: int


def run_holdout_eval(
    env: RemoteEnvClient,
    consumer: PolicyModel,
    baseline: BaselineJudge,
    seeds: list[int],
    stop_checker: Callable[[], bool] | None = None,
) -> EvalSummary:
    rewards: list[float] = []
    safe_rates: list[float] = []
    invalid_rates: list[float] = []
    baseline_scores: list[float] = []
    for seed in seeds:
        if stop_checker and stop_checker():
            break
        result = run_consumer_episode(env, consumer, {"task_id": "custom_episode", "seed": seed}, max_steps=24)
        rewards.append(result.total_reward)
        safe_rates.append(1.0 if result.safe_completion else 0.0)
        invalid_rates.append(result.invalid_action_rate)
        baseline_scores.append(baseline.score_episode(result.trace.get("state", {})))
    return EvalSummary(
        mean_reward=mean(rewards) if rewards else 0.0,
        safe_rate=mean(safe_rates) if safe_rates else 0.0,
        invalid_rate=mean(invalid_rates) if invalid_rates else 0.0,
        baseline_score=mean(baseline_scores) if baseline_scores else 0.0,
        episodes=len(seeds),
    )
