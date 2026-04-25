"""Typed models for DarkGuard OpenEnv interactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import ConfigDict, Field


class ActionType(str, Enum):
    INSPECT = "inspect"
    CLICK = "click"
    TOGGLE = "toggle"
    FLAG = "flag"
    GO_BACK = "go_back"
    SUBMIT = "submit"
    INVALID = "invalid"


@dataclass(slots=True)
class UIElement:
    id: str
    type: str
    text: str
    checked: bool = False
    enabled: bool = True
    prominence: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DarkGuardAction:
    action_type: ActionType
    target_id: str | None = None
    flag_category: str | None = None
    notes: str | None = None
    raw_text: str | None = None
    parser_error: str | None = None


@dataclass(slots=True)
class RewardBreakdown:
    progress: float = 0.0
    correct_flag: float = 0.0
    false_positive: float = 0.0
    harmful_path: float = 0.0
    safe_completion: float = 0.0
    efficiency: float = 0.0
    loop_penalty: float = 0.0
    invalid_action: float = 0.0
    evidence_bonus: float = 0.0

    def total(self) -> float:
        return (
            self.progress
            + self.correct_flag
            + self.false_positive
            + self.harmful_path
            + self.safe_completion
            + self.efficiency
            + self.loop_penalty
            + self.invalid_action
            + self.evidence_bonus
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "correct_flag": self.correct_flag,
            "false_positive": self.false_positive,
            "harmful_path": self.harmful_path,
            "safe_completion": self.safe_completion,
            "efficiency": self.efficiency,
            "loop_penalty": self.loop_penalty,
            "invalid_action": self.invalid_action,
            "evidence_bonus": self.evidence_bonus,
            "total": self.total(),
        }


@dataclass(slots=True)
class DarkGuardObservation:
    episode_id: str
    task_id: str
    screen_id: str
    instruction: str
    visible_summary: str
    elements: list[UIElement]
    allowed_actions: list[str]
    step_count: int
    max_steps: int
    reward: float
    cumulative_reward: float
    done: bool
    last_action_result: str
    messages: list[str]
    reward_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "screen_id": self.screen_id,
            "instruction": self.instruction,
            "visible_summary": self.visible_summary,
            "elements": [asdict(e) for e in self.elements],
            "allowed_actions": self.allowed_actions,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "reward": self.reward,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "last_action_result": self.last_action_result,
            "messages": self.messages,
            "reward_breakdown": self.reward_breakdown,
        }


@dataclass(slots=True)
class DarkGuardState:
    episode_id: str
    task_id: str
    screen_id: str
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool
    outcome_summary: str
    messages: list[str]
    reward_totals: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "screen_id": self.screen_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "outcome_summary": self.outcome_summary,
            "messages": self.messages,
            "reward_totals": self.reward_totals,
        }


class DarkGuardOpenEnvAction(Action):
    """OpenEnv Action model used by server/web interface."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(..., description="inspect|click|toggle|flag|go_back|submit")
    target_id: str | None = None
    flag_category: str | None = None
    notes: str | None = None


class DarkGuardOpenEnvObservation(Observation):
    """OpenEnv Observation model used by server/web interface."""

    episode_id: str
    task_id: str
    screen_id: str
    instruction: str
    visible_summary: str
    elements: list[dict[str, Any]]
    allowed_actions: list[str]
    step_count: int
    max_steps: int
    cumulative_reward: float
    last_action_result: str
    messages: list[str]
    reward_breakdown: dict[str, float] = Field(default_factory=dict)


class DarkGuardOpenEnvState(State):
    """OpenEnv State model used by server/web interface."""

    task_id: str
    screen_id: str
    max_steps: int
    cumulative_reward: float
    done: bool
    outcome_summary: str
    messages: list[str]
    reward_totals: dict[str, float]
