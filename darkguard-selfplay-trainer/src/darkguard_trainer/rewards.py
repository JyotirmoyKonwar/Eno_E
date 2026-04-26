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
    # The env_reward already includes breakdown components (fp, invalid, loop, efficiency).
    # To avoid double-counting, we extract the base env_reward and only add our aux parse_bonus.
    env_reward = float(obs.get("reward", 0.0))
    rb = obs.get("reward_breakdown", {})
    
    # We keep the pack for logging/visibility but set internal components to 0 
    # if they are already baked into env_reward.
    parse_bonus = 0.03 if action.get("action_type") else -0.05
    
    return RewardPack(
        env_reward=env_reward,
        safe_bonus=0.0,
        fp_penalty=0.0,
        invalid_penalty=0.0,
        loop_penalty=0.0,
        efficiency=0.0,
        parse_bonus=parse_bonus
    )


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
