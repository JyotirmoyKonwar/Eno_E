"""Model loading abstractions with lightweight fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAFE_HINTS = ("safe", "continue", "clean", "decline", "reject", "skip", "no thanks")
RISK_HINTS = ("accept all", "allow all", "marketing", "subscribe", "trap", "harm")


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
        last_result = str(observation.get("last_action_result", "")).lower()
        by_id = {str(e.get("id", "")): e for e in elements if isinstance(e, dict)}

        # Escape repeated dead-ends instead of re-clicking the same bad path.
        if "go_back" in allowed and ("invalid" in last_result or "no visible transition" in last_result):
            return {"action_type": "go_back"}

        # Prioritize inspection early to build state; this helps avoid fail-fast clicking.
        if "inspect" in allowed and elements:
            target = elements[0]["id"]
            return {"action_type": "inspect", "target_id": target, "notes": f"{self.role}-inspect"}

        if "click" in allowed and elements:
            # Score clickable candidates: prefer safe-looking labels and avoid risky ones.
            best_id = None
            best_score = float("-inf")
            for e in elements:
                target = str(e.get("id", ""))
                text = str(e.get("text", "")).lower()
                kind = str(e.get("type", "")).lower()
                if not target:
                    continue
                if kind not in {"button", "link", "checkbox", "toggle"}:
                    continue
                score = 0.0
                if any(h in text for h in SAFE_HINTS):
                    score += 2.0
                if any(h in text for h in RISK_HINTS):
                    score -= 2.0
                if e.get("checked") is True:
                    score -= 0.3
                score += 0.2 * self.skill_bias
                if score > best_score:
                    best_score = score
                    best_id = target
            target = best_id or str(elements[0].get("id", "")) or str(next(iter(by_id), ""))
            return {"action_type": "click", "target_id": target}
        return {"action_type": "submit"}

    def improve(self, delta: float) -> None:
        self.skill_bias = max(-2.0, min(2.0, self.skill_bias + delta))


def load_policy(role: str, base_model: str, adapter_repo: str, checkpoint_override: str = "") -> PolicyModel:
    return PolicyModel(
        role=role,
        base_model=base_model,
        adapter_repo=adapter_repo,
        checkpoint_path=checkpoint_override,
    )
