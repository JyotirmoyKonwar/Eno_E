"""Model loading abstractions with lightweight fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch
import random

SAFE_HINTS = ("safe", "continue", "clean", "decline", "reject", "skip", "no thanks")
RISK_HINTS = ("accept all", "allow all", "marketing", "subscribe", "trap", "harm")


@dataclass(slots=True)
class PolicyModel:
    role: str
    base_model: str
    adapter_repo: str
    checkpoint_path: str = ""
    skill_bias: float = 0.0
    action_logits: dict[str, float] = field(default_factory=lambda: {"click": 0.5, "flag": 0.3, "inspect": -0.1, "submit": 0.1, "go_back": -0.2})
    policy_lr: float = 0.05
    _episode_inspected: dict[str, set[str]] = field(default_factory=dict, repr=False)
    _traj_data: list[tuple[int, torch.Tensor]] = field(default_factory=list, repr=False)
    _traj_rewards: list[float] = field(default_factory=list, repr=False)

    def _get_probs(self) -> torch.Tensor:
        logits = torch.tensor([self.action_logits.get(k, 0.0) for k in ("click", "flag", "inspect", "submit", "go_back")], dtype=torch.float32)
        return torch.softmax(logits, dim=0)

    def _record_choice(self, action_type: str) -> None:
        idx_map = {"click": 0, "flag": 1, "inspect": 2, "submit": 3, "go_back": 4}
        idx = idx_map.get(action_type, 3)
        probs = self._get_probs()
        logprob = torch.log(probs[idx] + 1e-8)
        self._traj_data.append((idx, logprob.detach()))

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        elements = observation.get("elements", [])
        allowed = observation.get("allowed_actions", [])
        last_result = str(observation.get("last_action_result", "")).lower()
        inspected = self._seen_inspects(observation)
        
        # Stochastic exploration: 15% of the time, follow the policy distribution
        if random.random() < 0.15 and allowed:
            probs = self._get_probs()
            idx_map = ["click", "flag", "inspect", "submit", "go_back"]
            mask = torch.tensor([1.0 if k in allowed else 0.0 for k in idx_map])
            masked_probs = probs * mask
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
                choice_idx = int(torch.multinomial(masked_probs, 1).item())
                action_type = idx_map[choice_idx]
                self._record_choice(action_type)
                target_id = elements[0]["id"] if elements and action_type in {"click", "flag", "inspect"} else None
                return {"action_type": action_type, "target_id": target_id}

        # Heuristic fallback
        if "flag" in allowed:
            for e in elements:
                target = str(e.get("id", ""))
                if target and any(h in str(e.get("text","")).lower() for h in RISK_HINTS):
                    self._record_choice("flag")
                    return {"action_type": "flag", "target_id": target, "flag_category": "dark-pattern"}

        if "click" in allowed or "toggle" in allowed:
            best_id, best_score = None, float("-inf")
            for e in elements:
                target, text, kind = str(e.get("id", "")), str(e.get("text", "")).lower(), str(e.get("type", "")).lower()
                if not target or kind not in {"button", "link", "checkbox", "toggle"}: continue
                score = 2.0 if any(h in text for h in SAFE_HINTS) else (-1.0 if any(h in text for h in RISK_HINTS) else 0.0)
                score += 0.2 * self.skill_bias
                if score > best_score: best_id, best_score = target, score
            if best_id:
                atype = "click" if "click" in allowed else "toggle"
                self._record_choice("click")
                return {"action_type": atype, "target_id": best_id}

        if "inspect" in allowed:
            for e in elements:
                tid = str(e.get("id",""))
                if tid and tid not in inspected:
                    inspected.add(tid)
                    self._record_choice("inspect")
                    return {"action_type": "inspect", "target_id": tid}

        self._record_choice("submit")
        return {"action_type": "submit"}

    def ppo_update(self, clip_eps: float = 0.2) -> float:
        if not self._traj_data or not self._traj_rewards: return 0.0
        n = min(len(self._traj_data), len(self._traj_rewards))
        indices = [d[0] for d in self._traj_data[-n:]]
        old_logprobs = torch.stack([d[1] for d in self._traj_data[-n:]])
        rewards = torch.tensor(self._traj_rewards[-n:], dtype=torch.float32)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8) if len(rewards) > 1 else rewards
        
        opt_logits = torch.tensor([self.action_logits.get(k, 0.0) for k in ("click", "flag", "inspect", "submit", "go_back")], 
                                  dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([opt_logits], lr=self.policy_lr)
        
        for _ in range(3):
            optimizer.zero_grad()
            probs = torch.softmax(opt_logits, dim=0)
            curr_logprobs = torch.log(torch.stack([probs[i] for i in indices]) + 1e-8)
            ratio = torch.exp(curr_logprobs - old_logprobs)
            surr1, surr2 = ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            loss = -torch.min(surr1, surr2).mean()
            loss.backward(); optimizer.step()

        with torch.no_grad():
            for i, k in enumerate(("click", "flag", "inspect", "submit", "go_back")):
                self.action_logits[k] = float(opt_logits[i].item())
        self._traj_data.clear(); self._traj_rewards.clear()
        return float(loss.item())

    def design_episode(self, task_prompt: str, rng: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self._record_choice("click")
        tp = str(task_prompt).lower()
        knob = "cancel" if "cancel" in tp else ("checkout" if "checkout" in tp else "signup")
        subtlety = int(max(1, min(4, round(2 + self.skill_bias + rng.uniform(-0.8, 0.8)))))
        trap_id, safe_id = f"{knob}_trap", f"{knob}_safe"
        cfg = {
            "task_id": "custom_episode",
            "instruction": f"Complete {knob} safely.",
            "start_screen_id": "start",
            "screens": [
                {
                    "screen_id": "start",
                    "elements": [
                        {"id": trap_id, "type": "button", "text": "Accept all" if subtlety >= 3 else "Continue"},
                        {"id": safe_id, "type": "button", "text": "Decline extras"},
                    ],
                    "transitions": {trap_id: "harm_end", safe_id: "safe_end"},
                    "terminal": False,
                },
                {"screen_id": "safe_end", "elements": [{"id": "done", "type": "label", "text": "done"}], "terminal": True},
                {"screen_id": "harm_end", "elements": [{"id": "done", "type": "label", "text": "done"}], "terminal": True},
            ],
            "safe_terminal_ids": ["safe_end"], "harmful_terminal_ids": ["harm_end"],
            "trap_map": {trap_id: {"category": f"{knob}-pattern", "subtlety": subtlety}},
        }
        return cfg, {"role": self.role, "knob": knob, "subtlety": subtlety}

    def _episode_key(self, observation: dict[str, Any]) -> str:
        return str(observation.get("episode_id") or "default")

    def _seen_inspects(self, observation: dict[str, Any]) -> set[str]:
        key = self._episode_key(observation)
        if key not in self._episode_inspected:
            if len(self._episode_inspected) > 128: self._episode_inspected.clear()
            self._episode_inspected[key] = set()
        return self._episode_inspected[key]

    def improve(self, delta: float) -> None:
        self.skill_bias = max(-2.0, min(2.0, self.skill_bias + delta))

    def record_reward(self, reward: float) -> None:
        self._traj_rewards.append(float(reward))


def load_policy(role: str, base_model: str, adapter_repo: str, checkpoint_override: str = "") -> PolicyModel:
    return PolicyModel(role=role, base_model=base_model, adapter_repo=adapter_repo, checkpoint_path=checkpoint_override)
