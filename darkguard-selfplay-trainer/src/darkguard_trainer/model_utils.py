"""Model loading abstractions with lightweight fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .parser_utils import parse_action_text

VALID_ACTIONS = {"toggle", "inspect", "go_back", "flag", "submit", "click"}


@dataclass(slots=True)
class PolicyModel:
    role: str
    base_model: str
    adapter_repo: str
    checkpoint_path: str = ""
    skill_bias: float = 0.0
    router_enabled: bool = False
    _generator: Any | None = None

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        analysis = self.detect_dark_pattern(observation)
        action = self.route_action(analysis, observation)
        return self._sanitize_action(action, observation)

    def improve(self, delta: float) -> None:
        self.skill_bias = max(-2.0, min(2.0, self.skill_bias + delta))

    def detect_dark_pattern(self, observation: dict[str, Any]) -> str:
        """Cheap dark-pattern analysis surrogate (replaceable with model output)."""
        summary = str(observation.get("visible_summary", "")).lower()
        messages = " ".join(observation.get("messages", [])).lower()
        text = f"{summary} {messages}"
        if any(k in text for k in ["prechecked", "consent", "checkbox", "auto renew", "subscription"]):
            return "Detected preselected consent or recurring-charge dark pattern."
        if any(k in text for k in ["hidden", "small link", "friction", "cancel"]):
            return "Detected cancellation friction / roach motel pattern."
        if any(k in text for k in ["discount", "limited", "hurry", "countdown", "scarcity"]):
            return "Detected urgency or misleading discount pattern."
        return "No obvious dark pattern. Proceed carefully."

    def route_action(self, sft_output: str, observation: dict[str, Any]) -> dict[str, Any]:
        if self.router_enabled:
            generated = self._route_with_local_model(sft_output, observation)
            if generated:
                return generated
        return self._route_heuristic(sft_output, observation)

    def _route_with_local_model(self, sft_output: str, observation: dict[str, Any]) -> dict[str, Any] | None:
        try:
            generator = self._get_generator()
        except Exception:
            return None
        if generator is None:
            return None

        elem = self._pick_element(observation)
        user_prompt = (
            "Return only JSON action.\n"
            f"analysis: {sft_output}\n"
            f"element_id: {elem.get('id','')}\n"
            f"element_type: {elem.get('type','')}\n"
            f"element_text: {elem.get('text','')}\n"
            "Format: {\"action_type\":\"toggle|inspect|go_back|flag|submit|click\","
            "\"target_id\":\"optional\",\"notes\":\"required only for flag\"}\n"
        )
        result = generator(
            user_prompt,
            max_new_tokens=96,
            do_sample=False,
            return_full_text=False,
        )
        text = result[0]["generated_text"] if result else ""
        return parse_action_text(text)

    def _get_generator(self) -> Any | None:
        if self._generator is not None:
            return self._generator
        if os.getenv("ENABLE_LOCAL_ACTION_ROUTER", "").lower() not in {"1", "true", "yes"}:
            return None
        try:
            from transformers import pipeline

            self._generator = pipeline(
                "text-generation",
                model=self.base_model,
                tokenizer=self.base_model,
            )
            return self._generator
        except Exception:
            return None

    def _route_heuristic(self, sft_output: str, observation: dict[str, Any]) -> dict[str, Any]:
        text = (sft_output or "").lower()
        elem = self._pick_element(observation)
        eid = elem.get("id")
        if any(k in text for k in ["preselected", "prechecked", "opt-in", "checkbox"]):
            return {"action_type": "toggle", "target_id": eid}
        if any(k in text for k in ["obfuscated", "misleading", "hidden", "ambiguous"]):
            return {"action_type": "inspect", "target_id": eid}
        if any(k in text for k in ["roach motel", "cannot cancel", "hard to exit", "obstruction"]):
            return {"action_type": "go_back"}
        if any(k in text for k in ["urgency", "scarcity", "fake social proof", "bait-and-switch", "dark pattern"]):
            return {"action_type": "flag", "target_id": eid, "notes": "Detected suspected dark pattern from analysis."}
        if "safe" in text or "no obvious dark pattern" in text:
            return {"action_type": "submit", "target_id": eid}
        return {"action_type": "click", "target_id": eid}

    @staticmethod
    def _pick_element(observation: dict[str, Any]) -> dict[str, Any]:
        elements = observation.get("elements", [])
        if not elements:
            return {}
        # prefer actionable controls over static labels
        for element in elements:
            if element.get("type") in {"button", "checkbox", "toggle", "link"}:
                return element
        return elements[0]

    @staticmethod
    def _sanitize_action(action: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        allowed = set(observation.get("allowed_actions", [])) or VALID_ACTIONS
        elements = {e.get("id") for e in observation.get("elements", []) if e.get("id")}
        action_type = str(action.get("action_type", "click")).lower()
        if action_type not in VALID_ACTIONS or action_type not in allowed:
            action_type = "click" if "click" in allowed else next(iter(allowed))
        target_id = action.get("target_id") or action.get("element_id")
        if target_id and target_id not in elements:
            target_id = None
        if not target_id and elements and action_type in {"toggle", "inspect", "click", "flag", "submit"}:
            target_id = next(iter(elements))
        out = {"action_type": action_type}
        if target_id:
            out["target_id"] = target_id
        if action_type == "flag":
            out["notes"] = str(action.get("notes") or "Suspicious dark pattern signal detected.")
        return out


def load_policy(
    role: str,
    base_model: str,
    adapter_repo: str,
    checkpoint_override: str = "",
    *,
    router_enabled: bool = False,
) -> PolicyModel:
    return PolicyModel(
        role=role,
        base_model=base_model,
        adapter_repo=adapter_repo,
        checkpoint_path=checkpoint_override,
        router_enabled=router_enabled,
    )
