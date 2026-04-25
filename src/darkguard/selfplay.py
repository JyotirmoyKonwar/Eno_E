"""
DarkGuard self-play utilities.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple


def generate_designer_actions(subtlety: int, rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    """
    Generate a compact designer action program for one episode.
    """
    rng = rng or random.Random()
    subtlety = max(1, min(5, subtlety))
    actions: List[Dict[str, Any]] = []

    actions.append({"action": "set_price_discrepancy", "amount": 25 + 10 * subtlety})
    if subtlety >= 2:
        actions.append({"action": "obscure_label", "target": "auto_renew_checkbox"})
    if subtlety >= 3:
        actions.append({"action": "place_trap", "target": "seat_insurance_toggle"})
    if subtlety >= 4:
        actions.append({"action": "add_friction_screen", "edge": "cart_to_seat_selection"})
    if subtlety >= 5:
        actions.append({"action": "add_friction_screen", "edge": "cancellation_survey_to_final_confirmation"})

    rng.shuffle(actions)
    return actions


def apply_designer_actions(
    task_id: str,
    episode_config: Dict[str, Any],
    element_states: Dict[str, Any],
    designer_actions: List[Dict[str, Any]],
    subtlety: int,
) -> Dict[str, Any]:
    """
    Apply designer actions and return runtime modifiers consumed by env.
    """
    subtlety = max(1, min(5, subtlety))
    runtime = {
        "label_overrides": {},
        "friction_edges": set(),
        "price_delta": 0,
        "designer_actions_applied": [],
    }

    traps = {t.get("id") for t in episode_config.get("traps", [])}

    for action in designer_actions:
        kind = action.get("action")
        if kind == "set_price_discrepancy":
            delta = int(action.get("amount", 25))
            runtime["price_delta"] = max(0, min(100, delta))
            runtime["designer_actions_applied"].append({"action": kind, "amount": runtime["price_delta"]})
        elif kind == "obscure_label":
            target = action.get("target", "auto_renew_checkbox")
            runtime["label_overrides"][target] = "Keep convenience settings enabled"
            runtime["designer_actions_applied"].append({"action": kind, "target": target})
        elif kind == "place_trap":
            target = action.get("target", "seat_insurance_toggle")
            element_states[target] = True
            if target not in traps:
                episode_config.setdefault("traps", []).append(
                    {"id": target, "type": "designer_injected", "harmful": True}
                )
            runtime["designer_actions_applied"].append({"action": kind, "target": target})
        elif kind == "add_friction_screen":
            edge = action.get("edge", "cart_to_seat_selection")
            runtime["friction_edges"].add(edge)
            runtime["designer_actions_applied"].append({"action": kind, "edge": edge})

    return runtime


def update_elo(
    consumer_elo: float,
    designer_elo: float,
    consumer_score: float,
    k_factor: float = 24.0,
) -> Tuple[float, float]:
    """
    Two-player ELO update.
    consumer_score: 1.0 means consumer wins, 0.0 means designer wins.
    """
    ec = 1.0 / (1.0 + 10 ** ((designer_elo - consumer_elo) / 400.0))
    ed = 1.0 - ec
    dc = k_factor * (consumer_score - ec)
    dd = k_factor * ((1.0 - consumer_score) - ed)
    return consumer_elo + dc, designer_elo + dd
