"""Reward logic and anti-reward-hacking constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import DarkGuardAction, RewardBreakdown

FINAL_REWARD_MIN = -2.0
FINAL_REWARD_MAX = 2.0


@dataclass(slots=True)
class RewardContext:
    terminal_safe: bool = False
    terminal_harmful: bool = False
    was_invalid: bool = False
    inspected_new_target: bool = False
    repeated_state: bool = False
    false_flag: bool = False
    correct_flag: bool = False
    evidence_match: bool = False
    step_count: int = 0
    max_steps: int = 25


@dataclass(slots=True)
class RewardAccumulator:
    totals: RewardBreakdown = field(default_factory=RewardBreakdown)

    def update(self, action: DarkGuardAction, ctx: RewardContext) -> RewardBreakdown:
        rb = RewardBreakdown()

        if ctx.inspected_new_target:
            rb.progress += 0.08
        if action.action_type.value in {"click", "toggle", "go_back"}:
            rb.progress += 0.03

        if ctx.correct_flag:
            rb.correct_flag += 0.4
        if ctx.false_flag:
            rb.false_positive -= 0.25
        if ctx.evidence_match:
            rb.evidence_bonus += 0.08

        if ctx.terminal_harmful:
            rb.harmful_path -= 1.2
        if ctx.terminal_safe:
            rb.safe_completion += 3.0

        rb.efficiency -= 0.015
        if ctx.step_count > int(ctx.max_steps * 0.7):
            rb.efficiency -= 0.02

        if ctx.repeated_state:
            rb.loop_penalty -= 0.01

        if ctx.was_invalid:
            rb.invalid_action -= 0.02

        self._add_to_totals(rb)
        return rb

    def _add_to_totals(self, rb: RewardBreakdown) -> None:
        self.totals.progress += rb.progress
        self.totals.correct_flag += rb.correct_flag
        self.totals.false_positive += rb.false_positive
        self.totals.harmful_path += rb.harmful_path
        self.totals.safe_completion += rb.safe_completion
        self.totals.efficiency += rb.efficiency
        self.totals.loop_penalty += rb.loop_penalty
        self.totals.invalid_action += rb.invalid_action
        self.totals.evidence_bonus += rb.evidence_bonus


def clip_reward(value: float) -> float:
    return max(FINAL_REWARD_MIN, min(FINAL_REWARD_MAX, value))
