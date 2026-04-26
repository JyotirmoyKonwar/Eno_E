"""Model loading abstractions with lightweight fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONSUMER_SYSTEM_PROMPT = (
    "IMPORTANT: You MUST output valid JSON. "
    "If you cannot find the target, use ACTION: go_back or ACTION: inspect with a valid ID. "
    "Do NOT repeat failed actions."
)


@dataclass(slots=True)
class PolicyModel:
    role: str
    base_model: str
    adapter_repo: str
    checkpoint_path: str = ""
    skill_bias: float = 0.0

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        elements = observation.get("elements", [])
        allowed = observation.get("allowed_actions", [])
        if "inspect" in allowed and elements:
            target = elements[0]["id"]
            return {"action_type": "inspect", "target_id": target, "notes": f"{self.role}-inspect"}
        if "click" in allowed and elements:
            target = elements[-1]["id"]
            return {"action_type": "click", "target_id": target}
        return {"action_type": "submit"}

    def improve(self, delta: float) -> None:
        self.skill_bias = max(-2.0, min(2.0, self.skill_bias + delta))


def load_policy(role: str, base_model: str, adapter_repo: str, checkpoint_override: str = "") -> PolicyModel:
    _ = CONSUMER_SYSTEM_PROMPT if role == "consumer" else ""
    return PolicyModel(
        role=role,
        base_model=base_model,
        adapter_repo=adapter_repo,
        checkpoint_path=checkpoint_override,
    )
