"""
DarkGuard trap oracle.

This module provides a thin oracle layer that can be replaced by the
project's external classifier without touching environment logic.
"""

from typing import Any, Dict, Optional


class DarkGuardOracle:
    """
    Oracle used to evaluate whether an element is a trap.

    Current default behavior uses episode ground-truth trap ids as labels,
    while still exposing a score/reason interface that can be backed by a
    learned classifier later.
    """

    def evaluate_flag(
        self,
        episode_config: Dict[str, Any],
        element_id: Optional[str],
        element_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not element_id:
            return {"is_trap": False, "confidence": 0.0, "reason": "missing_element_id"}

        trap_ids = {trap.get("id") for trap in episode_config.get("traps", [])}
        is_trap = element_id in trap_ids

        # Confidence keeps interface stable for future classifier-backed oracle.
        confidence = 0.95 if is_trap else 0.1
        reason = "ground_truth_trap_match" if is_trap else "no_trap_match"

        # Lightweight metadata hinting improves interpretability for logs.
        if not is_trap and isinstance(element_metadata, dict):
            if any(k in element_metadata for k in ("type", "description", "amount")):
                confidence = 0.2
                reason = "metadata_contains_risk_like_fields_but_not_ground_truth_trap"

        return {"is_trap": is_trap, "confidence": confidence, "reason": reason}
