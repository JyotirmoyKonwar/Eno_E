"""
DarkGuard Step-Level Shaped Reward Computation.

All rewards are clipped to [-0.20, +0.20] per step.
This module is a pure function — no side effects.
"""

from typing import Any, Dict, List, Optional


# Reward constants
R_INSPECT_REVEALS       =  0.05   # inspect() found hidden harmful metadata
R_CROSS_SCREEN_COMPARE  =  0.08   # agent linked price/state discrepancy across screens
R_PROTECTIVE_ACTION     =  0.12   # agent took action to neutralise a trap before harm
R_GOAL_PROGRESS         =  0.06   # advancing toward goal (navigating correct path)
R_REPEAT_NOOP           = -0.03   # identical action twice in a row
R_SUBMIT_UNFLAGGED_TRAP = -0.05   # submitted while suspicious unflagged element exists
R_HARMFUL_STATE         = -0.10   # entered a harmful account state (charge, unwanted renew)
R_BAD_FLAG              = -0.08   # flagged a benign element
R_GOOD_FLAG             =  0.12   # correctly flagged a true trap
R_SUBMIT_WITHOUT_EVIDENCE = -0.10 # submitted without any inspect evidence

STEP_CLIP_HIGH =  0.20
STEP_CLIP_LOW  = -0.20

# Elements that are definitely benign — flagging these is punished
BENIGN_ELEMENTS = {
    "plan_free_trial", "plan_premium", "add_to_cart_btn",
    "proceed_btn", "continue_btn", "pay_btn",
    "help_btn", "upgrade_btn", "my_orders_link",
    "profile_section", "notifications_section", "privacy_link",
    "invoice_history_link", "payment_method_section", "current_plan_text",
    "downgrade_link", "pause_btn", "reason_dropdown",
    "additional_feedback_input", "submit_survey_btn", "skip_survey_link",
    "keep_benefits_btn", "name_input", "email_input", "password_input",
    "cart_item", "cart_total", "event_info", "offer_text", "current_plan_label",
    "seat_A1", "seat_B5", "seat_C12", "back_btn", "back_to_event_btn",
    "ticket_price_line", "account_settings_link", "billing_link",
    "manage_plan_link", "cancel_link",
}

# Elements that are genuine traps
TRAP_ELEMENTS = {
    "auto_renew_checkbox",
    "seat_insurance_toggle",
    "platform_fee_line",
    "dont_care_btn",
    "countdown_timer",
    "more_options_link",  # not harmful but revealing
}


def compute_step_reward(
    action_type: str,
    element_id: Optional[str],
    prev_screen_id: str,
    next_screen_id: str,
    account_state: Dict[str, Any],
    prev_account_state: Dict[str, Any],
    task_id: str,
    episode_config: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    element_states: Dict[str, Any],
    inspected_elements: List[str],
    flags_submitted: List[Dict[str, Any]],
    inspect_result: Optional[Dict[str, Any]] = None,
    flag_note: Optional[str] = None,
    flag_verdict: Optional[Dict[str, Any]] = None,
    inspected_before_submit: bool = False,
    is_terminal_submit: bool = False,
) -> float:
    """
    Compute the shaped step reward for the given transition.

    Returns a float clipped to [STEP_CLIP_LOW, STEP_CLIP_HIGH].
    """
    reward = 0.0

    # 1. inspect() reveals harmful hidden metadata → +0.05
    if action_type == "inspect" and inspect_result:
        trap_ids = {t["id"] for t in episode_config.get("traps", [])}
        if element_id in trap_ids:
            reward += R_INSPECT_REVEALS

    # 2. Cross-screen price/state comparison (medium task specific)
    # Agent inspected the advertised price AND the total_label on payment screen
    if task_id == "medium_fair_checkout" and action_type == "inspect":
        if element_id in ("price_label", "total_label"):
            both_inspected = (
                "price_label" in inspected_elements
                and "total_label" in inspected_elements
            )
            if both_inspected:
                # Only give this bonus once (first time both are inspected)
                prev_count = sum(
                    1 for h in action_history
                    if h.get("action_type") == "inspect"
                    and h.get("element_id") in ("price_label", "total_label")
                )
                if prev_count == 1:  # This is the second of the pair being inspected
                    reward += R_CROSS_SCREEN_COMPARE

    # 3. Protective action BEFORE irreversible harm
    if action_type == "toggle" and element_id == "auto_renew_checkbox":
        # toggling off the preselected checkbox = protective
        if not element_states.get("auto_renew_checkbox", True):
            reward += R_PROTECTIVE_ACTION

    if action_type == "toggle" and element_id == "seat_insurance_toggle":
        if not element_states.get("seat_insurance_toggle", True):
            reward += R_PROTECTIVE_ACTION

    # Going back from payment review (suspicious fee visible) is protective
    if action_type == "go_back" and prev_screen_id == "payment_review":
        reward += R_PROTECTIVE_ACTION * 0.5

    # Clicking dont_care_btn (correct button) on retention screen — brave move
    if action_type == "click" and element_id == "dont_care_btn":
        reward += R_PROTECTIVE_ACTION * 0.5

    # 4. Goal progress — navigating on the correct path
    correct_path_steps = {
        "easy_safe_signup": {("plan_selection", "signup_form")},
        "medium_fair_checkout": {
            ("event_page", "cart"),
            ("cart", "seat_selection"),
            ("seat_selection", "payment_review"),
        },
        "hard_cancel_maze": {
            ("account_home", "account_settings"),
            ("account_settings", "billing"),
            ("billing", "manage_plan"),
            ("manage_plan", "more_options"),
            ("more_options", "retention_offer"),
            ("retention_offer", "cancellation_survey"),
            ("cancellation_survey", "final_confirmation"),
        },
    }
    if (prev_screen_id, next_screen_id) in correct_path_steps.get(task_id, set()):
        reward += R_GOAL_PROGRESS

    # 5. Repeated no-op / identical action
    if len(action_history) >= 2:
        last = action_history[-1]
        second_last = action_history[-2]
        if (
            last.get("action_type") == second_last.get("action_type")
            and last.get("element_id") == second_last.get("element_id")
        ):
            reward += R_REPEAT_NOOP

    # 6. Submitted while a suspicious unflagged trap element is in harmful state
    if is_terminal_submit:
        trap_ids = {t["id"] for t in episode_config.get("traps", [])}
        flagged_ids = {f["element_id"] for f in flags_submitted}
        unflagged_harmful = _has_unflagged_harmful_trap(
            task_id, element_states, trap_ids, flagged_ids
        )
        if unflagged_harmful:
            reward += R_SUBMIT_UNFLAGGED_TRAP

    # 7. Entered harmful account state (compared to previous)
    reward += _harmful_state_delta(account_state, prev_account_state, task_id)

    # 8. Flag quality from oracle verdict
    if action_type == "flag":
        if flag_verdict is not None:
            if flag_verdict.get("is_trap", False):
                reward += R_GOOD_FLAG
            else:
                reward += R_BAD_FLAG
        elif element_id in BENIGN_ELEMENTS:
            # fallback if oracle verdict not passed
            reward += R_BAD_FLAG

    # 9. Overconfident submit (no evidence gathering first)
    if is_terminal_submit and not inspected_before_submit:
        reward += R_SUBMIT_WITHOUT_EVIDENCE

    return float(max(STEP_CLIP_LOW, min(STEP_CLIP_HIGH, reward)))


def _has_unflagged_harmful_trap(
    task_id: str,
    element_states: Dict[str, Any],
    trap_ids: set,
    flagged_ids: set,
) -> bool:
    """Check if any harmful trap is currently active and not yet flagged."""
    if task_id == "easy_safe_signup":
        return (
            element_states.get("auto_renew_checkbox", True)
            and "auto_renew_checkbox" not in flagged_ids
        )
    if task_id == "medium_fair_checkout":
        insurance_active = element_states.get("seat_insurance_toggle", True)
        return (
            insurance_active and "seat_insurance_toggle" not in flagged_ids
        ) or "platform_fee_line" not in flagged_ids
    return False


def _harmful_state_delta(
    account_state: Dict[str, Any],
    prev_account_state: Dict[str, Any],
    task_id: str,
) -> float:
    """Return penalty if the account entered a harmful state this step."""
    if task_id == "easy_safe_signup":
        if not prev_account_state.get("auto_renew") and account_state.get("auto_renew"):
            return R_HARMFUL_STATE
    if task_id == "medium_fair_checkout":
        prev_charge = prev_account_state.get("total_charged", 0)
        curr_charge = account_state.get("total_charged", 0)
        if curr_charge > 500 and prev_charge <= 500:
            return R_HARMFUL_STATE
    if task_id == "hard_cancel_maze":
        # Subscription remained active despite an attempt to cancel
        # (e.g. agent clicked Keep my benefits or got stuck looping)
        if (
            account_state.get("subscription_active", True)
            and prev_account_state.get("subscription_active", True)
            and prev_account_state.get("cancellation_confirmed", False) is False
            and account_state.get("cancellation_confirmed", False) is False
        ):
            # Penalise only when the agent just failed a potential final-confirm step
            # to avoid double-penalising every idle step.
            pass  # coarse loop penalty handled by R_REPEAT_NOOP above
        # Hard penalty: subscription stayed active after the agent thought it submitted
        if (
            account_state.get("subscription_active", True)
            and not prev_account_state.get("subscription_active", True)
        ):
            # Should never happen, but guard against state regression
            return R_HARMFUL_STATE
    return 0.0
