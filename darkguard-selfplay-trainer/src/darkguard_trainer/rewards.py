"""Reward routing for consumer and designer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RewardPack:
    env_reward: float
    safe_bonus: float
    fp_penalty: float
    invalid_penalty: float
    loop_penalty: float
    efficiency: float
    parse_bonus: float

    @property
    def total(self) -> float:
        return (
            self.env_reward
            + self.safe_bonus
            + self.fp_penalty
            + self.invalid_penalty
            + self.loop_penalty
            + self.efficiency
            + self.parse_bonus
        )


def compute_consumer_reward(obs: dict[str, Any], action: dict[str, Any]) -> RewardPack:
    env_reward = float(obs.get("reward", 0.0))
    safe_bonus = 0.1 if "safe" in str(obs.get("last_action_result", "")).lower() else 0.0
    # The environment reward already includes reward_breakdown components.
    # Keep these values for logging/debug visibility but do not add them again.
    fp_penalty = 0.0
    invalid_penalty = 0.0
    loop_penalty = 0.0
    efficiency = 0.0
    parse_bonus = 0.03 if action.get("action_type") else -0.03
    return RewardPack(env_reward, safe_bonus, fp_penalty, invalid_penalty, loop_penalty, efficiency, parse_bonus)


def compute_designer_reward(
    valid: bool,
    challenge_delta: float,
    novelty: float,
    leakage_detected: bool,
    impossible: bool,
) -> float:
    reward = 0.0
    reward += 0.4 if valid else -0.8
    reward += max(-0.5, min(0.8, challenge_delta))
    reward += max(-0.2, min(0.3, novelty))
    if leakage_detected:
        reward -= 0.5
    if impossible:
        reward -= 0.8
    return max(-2.0, min(2.0, reward))


def split_reward_components(total: float) -> dict[str, float]:
    return {
        "primary_env": total * 0.7,
        "aux_stability": total * 0.2,
        "format_guard": total * 0.1,
    }
