"""
DarkGuard — Determinism Tests.

Verifies that identical action sequences produce identical scores every run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.darkguard.env import DarkGuardEnv
from src.darkguard.models import DarkGuardAction


def run_easy_scenario(env: DarkGuardEnv) -> dict:
    """Run the safe signup scenario and return final metadata."""
    env.reset(task_id="easy_safe_signup")
    env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
    env.step(DarkGuardAction(action_type="inspect", element_id="auto_renew_checkbox"))
    env.step(DarkGuardAction(action_type="flag", element_id="auto_renew_checkbox", note="preselected harmful"))
    env.step(DarkGuardAction(action_type="toggle", element_id="auto_renew_checkbox"))
    obs = env.step(DarkGuardAction(action_type="submit", element_id="submit_btn"))
    return {
        "episode_score": obs.metadata.get("episode_score"),
        "account_state": obs.account_state.copy(),
        "cumulative_reward": obs.cumulative_reward,
        "done": obs.done,
    }


def run_medium_scenario(env: DarkGuardEnv) -> dict:
    env.reset(task_id="medium_fair_checkout")
    env.step(DarkGuardAction(action_type="inspect", element_id="price_label"))
    env.step(DarkGuardAction(action_type="click", element_id="add_to_cart_btn"))
    env.step(DarkGuardAction(action_type="click", element_id="proceed_btn"))
    env.step(DarkGuardAction(action_type="inspect", element_id="seat_insurance_toggle"))
    env.step(DarkGuardAction(action_type="toggle", element_id="seat_insurance_toggle"))
    env.step(DarkGuardAction(action_type="click", element_id="continue_btn"))
    env.step(DarkGuardAction(action_type="inspect", element_id="platform_fee_line"))
    env.step(DarkGuardAction(action_type="inspect", element_id="total_label"))
    env.step(DarkGuardAction(action_type="go_back", element_id=None))
    obs = env.step(DarkGuardAction(action_type="click", element_id="continue_btn"))
    return {
        "episode_score": obs.metadata.get("episode_score"),
        "account_state": obs.account_state.copy(),
        "done": obs.done,
    }


class TestDeterminism:
    def test_easy_same_sequence_same_score(self):
        env1 = DarkGuardEnv()
        env2 = DarkGuardEnv()
        result1 = run_easy_scenario(env1)
        result2 = run_easy_scenario(env2)
        assert result1["episode_score"] == result2["episode_score"], (
            f"Scores differ: {result1['episode_score']} vs {result2['episode_score']}"
        )
        assert result1["account_state"] == result2["account_state"]
        assert result1["cumulative_reward"] == result2["cumulative_reward"]
        assert result1["done"] is True
        assert result2["done"] is True

    def test_medium_same_sequence_same_score(self):
        env1 = DarkGuardEnv()
        env2 = DarkGuardEnv()
        result1 = run_medium_scenario(env1)
        result2 = run_medium_scenario(env2)
        assert result1["account_state"] == result2["account_state"]

    def test_different_sequences_different_outcomes(self):
        """Safe vs harmful path gives different scores."""
        env = DarkGuardEnv()

        # Safe path
        env.reset(task_id="easy_safe_signup")
        env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        env.step(DarkGuardAction(action_type="toggle", element_id="auto_renew_checkbox"))
        obs_safe = env.step(DarkGuardAction(action_type="submit", element_id="submit_btn"))

        # Harmful path
        env.reset(task_id="easy_safe_signup")
        env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        obs_harm = env.step(DarkGuardAction(action_type="submit", element_id="submit_btn"))

        safe_score = obs_safe.metadata.get("episode_score", 0)
        harm_score = obs_harm.metadata.get("episode_score", 0)
        assert safe_score > harm_score, (
            f"Safe score ({safe_score}) should exceed harmful score ({harm_score})"
        )

    def test_grader_pure_function_determinism(self):
        """Same inputs to grader always return same output."""
        from src.darkguard.grader import compute_episode_score

        gt = {
            "task_id": "easy_safe_signup",
            "max_steps": 15,
            "traps": [{"id": "auto_renew_checkbox"}],
            "relevant_elements_for_evidence": ["auto_renew_checkbox"],
        }
        trace = {
            "final_account_state": {"trial_started": True, "auto_renew": False},
            "flags_submitted": [{"element_id": "auto_renew_checkbox"}],
            "inspected_elements": ["auto_renew_checkbox"],
            "screens_visited": ["plan_selection", "signup_form"],
            "steps_taken": 5,
        }

        results = [compute_episode_score(gt, trace) for _ in range(10)]
        first = results[0]["episode_score"]
        assert all(r["episode_score"] == first for r in results), \
            "Grader is not deterministic!"

    def test_reset_clears_state(self):
        """reset() must fully clear episode state."""
        env = DarkGuardEnv()
        env.reset(task_id="easy_safe_signup")
        env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        env.step(DarkGuardAction(action_type="submit", element_id="submit_btn"))
        assert env.state.step_count > 0

        # Reset to new task
        env.reset(task_id="hard_cancel_maze")
        assert env.state.step_count == 0
        assert env.state.task_id == "hard_cancel_maze"
        assert env.state.account_state["subscription_active"] is True
