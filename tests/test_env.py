"""
DarkGuard — Environment Tests.

Tests the reset/step/state cycle for all 3 tasks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.darkguard.env import DarkGuardEnv
from src.darkguard.models import DarkGuardAction


# ── Easy task ────────────────────────────────────────────────────────────────

class TestEasyTask:
    def setup_method(self):
        self.env = DarkGuardEnv()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="easy_safe_signup")
        assert obs.task_id == "easy_safe_signup"
        assert obs.screen_id == "plan_selection"
        assert obs.done is False
        assert obs.step == 0
        assert len(obs.elements) > 0
        assert obs.user_goal != ""

    def test_initial_account_state(self):
        obs = self.env.reset(task_id="easy_safe_signup")
        assert obs.account_state["trial_started"] is False
        assert obs.account_state["auto_renew"] is False

    def test_auto_renew_preselected(self):
        """Auto-renew checkbox should be preselected (the trap)."""
        self.env.reset(task_id="easy_safe_signup")
        # Navigate to signup form
        obs = self.env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        checkbox = next((e for e in obs.elements if e["id"] == "auto_renew_checkbox"), None)
        assert checkbox is not None
        assert checkbox["selected"] is True

    def test_toggle_disables_auto_renew(self):
        self.env.reset(task_id="easy_safe_signup")
        self.env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        obs = self.env.step(DarkGuardAction(action_type="toggle", element_id="auto_renew_checkbox"))
        checkbox = next((e for e in obs.elements if e["id"] == "auto_renew_checkbox"), None)
        assert checkbox is not None
        assert checkbox["selected"] is False

    def test_inspect_reveals_metadata(self):
        self.env.reset(task_id="easy_safe_signup")
        self.env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        obs = self.env.step(DarkGuardAction(action_type="inspect", element_id="auto_renew_checkbox"))
        # inspect should give a positive or at least non-negative reward
        assert obs.step_reward >= 0.0

    def test_safe_completion(self):
        """Full safe path: disable auto-renew then submit."""
        self.env.reset(task_id="easy_safe_signup")
        self.env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        self.env.step(DarkGuardAction(action_type="inspect", element_id="auto_renew_checkbox"))
        self.env.step(DarkGuardAction(action_type="toggle", element_id="auto_renew_checkbox"))
        obs = self.env.step(DarkGuardAction(action_type="submit", element_id="submit_btn"))
        assert obs.done is True
        assert obs.account_state["trial_started"] is True
        assert obs.account_state["auto_renew"] is False
        score = obs.metadata.get("episode_score", 0)
        assert score >= 0.55  # P=1, G=1

    def test_harmful_completion(self):
        """Submit with auto-renew still on → harmful outcome."""
        self.env.reset(task_id="easy_safe_signup")
        self.env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        obs = self.env.step(DarkGuardAction(action_type="submit", element_id="submit_btn"))
        assert obs.done is True
        assert obs.account_state["auto_renew"] is True
        assert obs.account_state["charge_on_day_8"] == 499
        score = obs.metadata.get("episode_score", 1.0)
        assert score < 0.55  # P=0

    def test_state_returns_ground_truth(self):
        self.env.reset(task_id="easy_safe_signup")
        state = self.env.state
        assert "traps" in state.ground_truth
        assert state.task_id == "easy_safe_signup"


# ── Medium task ───────────────────────────────────────────────────────────────

class TestMediumTask:
    def setup_method(self):
        self.env = DarkGuardEnv()

    def test_reset(self):
        obs = self.env.reset(task_id="medium_fair_checkout")
        assert obs.screen_id == "event_page"
        assert obs.task_id == "medium_fair_checkout"

    def test_price_visible_on_event_page(self):
        obs = self.env.reset(task_id="medium_fair_checkout")
        price = next((e for e in obs.elements if e["id"] == "price_label"), None)
        assert price is not None
        assert "500" in price["label"]

    def test_insurance_preselected(self):
        self.env.reset(task_id="medium_fair_checkout")
        self.env.step(DarkGuardAction(action_type="click", element_id="add_to_cart_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="proceed_btn"))
        obs = self.env.step(DarkGuardAction(action_type="inspect", element_id="seat_insurance_toggle"))
        # insurance toggle should be on seat_selection screen
        toggle = next((e for e in obs.elements if e["id"] == "seat_insurance_toggle"), None)
        assert toggle is not None
        assert toggle["selected"] is True

    def test_platform_fee_appears_at_payment(self):
        self.env.reset(task_id="medium_fair_checkout")
        self.env.step(DarkGuardAction(action_type="click", element_id="add_to_cart_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="proceed_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="seat_A1"))
        obs = self.env.step(DarkGuardAction(action_type="click", element_id="continue_btn"))
        fee = next((e for e in obs.elements if e["id"] == "platform_fee_line"), None)
        assert fee is not None, "Platform fee must appear on payment review screen"

    def test_fair_checkout_score(self):
        """Disable insurance → still overpays platform fee → P=0.5."""
        self.env.reset(task_id="medium_fair_checkout")
        self.env.step(DarkGuardAction(action_type="click", element_id="add_to_cart_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="proceed_btn"))
        self.env.step(DarkGuardAction(action_type="toggle", element_id="seat_insurance_toggle"))
        self.env.step(DarkGuardAction(action_type="click", element_id="continue_btn"))
        obs = self.env.step(DarkGuardAction(action_type="click", element_id="pay_btn"))
        assert obs.done is True
        # Total should be 575 (no insurance, but platform fee still there)
        assert obs.account_state["total_charged"] == 575


# ── Hard task ─────────────────────────────────────────────────────────────────

class TestHardTask:
    def setup_method(self):
        self.env = DarkGuardEnv()

    def test_reset(self):
        obs = self.env.reset(task_id="hard_cancel_maze")
        assert obs.screen_id == "account_home"
        assert obs.account_state["subscription_active"] is True

    def test_navigate_to_billing(self):
        self.env.reset(task_id="hard_cancel_maze")
        self.env.step(DarkGuardAction(action_type="click", element_id="account_settings_link"))
        obs = self.env.step(DarkGuardAction(action_type="click", element_id="billing_link"))
        assert obs.screen_id == "billing"

    def test_full_cancel_path(self):
        """Navigate the full maze and cancel correctly."""
        self.env.reset(task_id="hard_cancel_maze")
        self.env.step(DarkGuardAction(action_type="click", element_id="account_settings_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="billing_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="manage_plan_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="more_options_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="cancel_link"))
        self.env.step(DarkGuardAction(action_type="inspect", element_id="dont_care_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="dont_care_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="submit_survey_btn"))
        self.env.step(DarkGuardAction(action_type="inspect", element_id="countdown_timer"))
        self.env.step(DarkGuardAction(action_type="click", element_id="confirm_cancel_btn"))
        obs = self.env.step(DarkGuardAction(action_type="click", element_id="final_confirm_btn"))
        assert obs.done is True
        assert obs.account_state["subscription_active"] is False

    def test_premature_submit_fails(self):
        """Submitting without first_confirm means subscription stays active."""
        self.env.reset(task_id="hard_cancel_maze")
        # Navigate to final_confirmation
        self.env.step(DarkGuardAction(action_type="click", element_id="account_settings_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="billing_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="manage_plan_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="more_options_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="cancel_link"))
        self.env.step(DarkGuardAction(action_type="click", element_id="dont_care_btn"))
        self.env.step(DarkGuardAction(action_type="click", element_id="submit_survey_btn"))
        # Submit without clicking confirm_cancel_btn first
        obs = self.env.step(DarkGuardAction(action_type="submit", element_id=None))
        # Should not have cancelled
        assert obs.account_state["subscription_active"] is True


# ── General ───────────────────────────────────────────────────────────────────

class TestGeneral:
    def test_max_steps_terminates_episode(self):
        env = DarkGuardEnv()
        env.reset(task_id="easy_safe_signup")
        for _ in range(15):
            obs = env.step(DarkGuardAction(action_type="go_back", element_id=None))
        assert obs.done is True

    def test_cumulative_reward_accumulates(self):
        env = DarkGuardEnv()
        env.reset(task_id="easy_safe_signup")
        env.step(DarkGuardAction(action_type="click", element_id="plan_free_trial"))
        obs = env.step(DarkGuardAction(action_type="inspect", element_id="auto_renew_checkbox"))
        assert obs.cumulative_reward != 0.0

    def test_observations_have_required_fields(self):
        env = DarkGuardEnv()
        obs = env.reset(task_id="medium_fair_checkout")
        assert hasattr(obs, "episode_id")
        assert hasattr(obs, "task_id")
        assert hasattr(obs, "screen_id")
        assert hasattr(obs, "elements")
        assert hasattr(obs, "account_state")
        assert hasattr(obs, "event_log")
        assert hasattr(obs, "cumulative_reward")
        assert hasattr(obs, "user_goal")
