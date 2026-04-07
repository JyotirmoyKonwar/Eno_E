"""
DarkGuard Environment — Main Environment Class.

Implements the OpenEnv Environment interface:
  reset(task_id=None) → DarkGuardObservation
  step(action)        → DarkGuardObservation
  state               → DarkGuardState
"""

import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .grader import compute_episode_score
from .models import DarkGuardAction, DarkGuardObservation, DarkGuardState
from .rewards import compute_step_reward
from .screens import (
    TASK_INITIAL_SCREENS,
    element_exists,
    get_element_metadata,
    get_elements_list,
    get_screen,
    get_transition,
)

# ---------------------------------------------------------------------------
# Episode config loader
# ---------------------------------------------------------------------------

_EPISODES_DIR = os.path.join(os.path.dirname(__file__), "episodes")

TASK_IDS = [
    "easy_safe_signup",
    "medium_fair_checkout",
    "hard_cancel_maze",
]


def _load_episode(task_id: str) -> Dict[str, Any]:
    path = os.path.join(_EPISODES_DIR, f"{task_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


TASK_GOALS = {
    "easy_safe_signup": "Start the free trial without enabling automatic paid renewal.",
    "medium_fair_checkout": "Buy one concert ticket and pay no more than the advertised price of ₹500.",
    "hard_cancel_maze": "Cancel the subscription completely. subscription_active must be False at the end.",
}

TASK_MAX_STEPS = {
    "easy_safe_signup": 15,
    "medium_fair_checkout": 25,
    "hard_cancel_maze": 40,
}

# Initial element toggle states (task-specific defaults)
TASK_INITIAL_ELEMENT_STATES: Dict[str, Dict[str, Any]] = {
    "easy_safe_signup": {
        "auto_renew_checkbox": True,   # preselected — the trap
    },
    "medium_fair_checkout": {
        "seat_insurance_toggle": True,  # preselected — the trap
        "selected_seat": None,
    },
    "hard_cancel_maze": {
        "survey_done": False,
        "first_confirm_clicked": False,
    },
}

TASK_INITIAL_ACCOUNT_STATES: Dict[str, Dict[str, Any]] = {
    "easy_safe_signup": {
        "trial_started": False,
        "auto_renew": False,
        "charge_on_day_8": 0,
    },
    "medium_fair_checkout": {
        "ticket_purchased": False,
        "total_charged": 0,
        "seat_insurance": False,
        "selected_seat": None,
    },
    "hard_cancel_maze": {
        "subscription_active": True,
        "survey_completed": False,
        "cancellation_confirmed": False,
    },
}


class DarkGuardEnv(Environment):
    """
    DarkGuard Consumer Protection RL Environment.

    Simulates 3 real-world digital consumer flows (free trial signup,
    product checkout, subscription cancellation) with embedded dark patterns.
    The agent must detect and neutralise manipulative traps while completing
    the user's goal.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        # Episode state — initialised in reset()
        self._episode_id: str = str(uuid4())
        self._task_id: str = "easy_safe_signup"
        self._episode_config: Dict[str, Any] = {}
        self._screen_id: str = ""
        self._screen_history: List[str] = []
        self._account_state: Dict[str, Any] = {}
        self._element_states: Dict[str, Any] = {}
        self._event_log: List[str] = []
        self._action_history: List[Dict[str, Any]] = []
        self._flags_submitted: List[Dict[str, Any]] = []
        self._inspected_elements: List[str] = []
        self._screens_visited: List[str] = []
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._episode_score: Optional[float] = None

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DarkGuardObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: One of 'easy_safe_signup', 'medium_fair_checkout',
                     'hard_cancel_maze'. If None, cycles through all 3.
        """
        # Resolve task
        if task_id is None:
            task_id = random.choice(TASK_IDS)
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {TASK_IDS}")

        self._task_id = task_id
        self._episode_id = episode_id or str(uuid4())
        self._episode_config = _load_episode(task_id)
        self._screen_id = TASK_INITIAL_SCREENS[task_id]
        self._screen_history = [self._screen_id]
        self._account_state = deepcopy(TASK_INITIAL_ACCOUNT_STATES[task_id])
        self._element_states = deepcopy(TASK_INITIAL_ELEMENT_STATES[task_id])
        self._event_log = [f"Episode started. Task: {task_id}"]
        self._action_history = []
        self._flags_submitted = []
        self._inspected_elements = []
        self._screens_visited = [self._screen_id]
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_score = None

        return self._build_observation(step_reward=0.0)

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    def step(
        self,
        action: DarkGuardAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DarkGuardObservation:
        """Apply one action and return the next observation."""
        if self._done:
            # Return terminal observation without advancing
            obs = self._build_observation(step_reward=0.0)
            obs.done = True
            return obs

        self._step_count += 1
        prev_screen_id = self._screen_id
        prev_account_state = deepcopy(self._account_state)

        action_type = action.action_type
        element_id = action.element_id

        # Record action before processing
        self._action_history.append({
            "step": self._step_count,
            "action_type": action_type,
            "element_id": element_id,
            "value": action.value,
            "note": action.note,
            "screen_id": self._screen_id,
        })

        # ---- Validate element exists (if element_id is given) ----
        inspect_result = None
        if element_id and action_type not in ("go_back", "submit", "flag"):
            if not element_exists(
                self._task_id, self._screen_id, element_id,
                self._account_state, self._element_states
            ):
                self._event_log.append(
                    f"Step {self._step_count}: Invalid action — element '{element_id}' not found on screen '{self._screen_id}'"
                )
                step_reward = -0.03  # small penalty for invalid actions
                self._cumulative_reward = round(self._cumulative_reward + step_reward, 4)
                return self._build_observation(step_reward=step_reward)

        # ---- Handle each action type ----
        if action_type == "inspect":
            inspect_result = self._handle_inspect(element_id)
        elif action_type == "toggle":
            self._handle_toggle(element_id)
        elif action_type == "type":
            self._handle_type(element_id, action.value)
        elif action_type == "flag":
            self._handle_flag(element_id, action.note)
        elif action_type == "go_back":
            self._handle_go_back()
        elif action_type in ("click", "submit"):
            self._handle_click_or_submit(action_type, element_id)

        # ---- Check max steps ----
        max_steps = TASK_MAX_STEPS[self._task_id]
        if self._step_count >= max_steps and not self._done:
            self._done = True
            self._event_log.append(f"Episode ended: max steps ({max_steps}) reached.")
            self._finalise_episode()

        # ---- Compute step reward ----
        is_terminal_submit = (action_type in ("submit", "click") and self._done)
        step_reward = compute_step_reward(
            action_type=action_type,
            element_id=element_id,
            prev_screen_id=prev_screen_id,
            next_screen_id=self._screen_id,
            account_state=self._account_state,
            prev_account_state=prev_account_state,
            task_id=self._task_id,
            episode_config=self._episode_config,
            action_history=self._action_history,
            element_states=self._element_states,
            inspected_elements=self._inspected_elements,
            flags_submitted=self._flags_submitted,
            inspect_result=inspect_result,
            flag_note=action.note,
            is_terminal_submit=is_terminal_submit,
        )
        self._cumulative_reward = round(self._cumulative_reward + step_reward, 4)
        self._event_log.append(
            f"Step {self._step_count}: {action_type}"
            + (f"({element_id})" if element_id else "")
            + f" → screen={self._screen_id} reward={step_reward:+.3f}"
        )

        obs = self._build_observation(step_reward=step_reward)
        return obs

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_inspect(self, element_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not element_id:
            return None
        metadata = get_element_metadata(
            self._task_id, self._screen_id, element_id,
            self._account_state, self._element_states
        )
        if element_id not in self._inspected_elements:
            self._inspected_elements.append(element_id)
        if metadata:
            self._event_log.append(
                f"  [INSPECT] {element_id}: {metadata}"
            )
        return metadata

    def _handle_toggle(self, element_id: Optional[str]) -> None:
        if not element_id:
            return
        current = self._element_states.get(element_id, False)
        self._element_states[element_id] = not current
        self._event_log.append(
            f"  [TOGGLE] {element_id}: {current} → {not current}"
        )

    def _handle_type(self, element_id: Optional[str], value: Optional[str]) -> None:
        if element_id:
            self._element_states[f"typed_{element_id}"] = value or ""

    def _handle_flag(self, element_id: Optional[str], note: Optional[str]) -> None:
        if not element_id:
            return
        self._flags_submitted.append({
            "element_id": element_id,
            "note": note,
            "screen_id": self._screen_id,
            "step": self._step_count,
        })
        self._event_log.append(f"  [FLAG] {element_id}: {note}")

    def _handle_go_back(self) -> None:
        if len(self._screen_history) > 1:
            self._screen_history.pop()
            self._screen_id = self._screen_history[-1]
        else:
            self._event_log.append("  [GO_BACK] Already at first screen — wasted step.")

    def _handle_click_or_submit(self, action_type: str, element_id: Optional[str]) -> None:
        # Look up transition
        next_screen = get_transition(
            self._task_id, self._screen_id, action_type, element_id,
            self._account_state, self._element_states
        )

        if next_screen is None:
            # No transition found — try "click" as fallback for submit
            if action_type == "submit":
                next_screen = get_transition(
                    self._task_id, self._screen_id, "click", element_id,
                    self._account_state, self._element_states
                )

        if next_screen == "__self__":
            return  # no screen change

        if next_screen == "__first_confirm__":
            # Hard task: first confirm click — reveal final_confirm_btn
            self._element_states["first_confirm_clicked"] = True
            self._event_log.append("  [CONFIRM] First confirm clicked. Final confirm button now visible.")
            return

        if next_screen == "__submit__" or (
            action_type == "submit" and next_screen is None
        ):
            self._finalise_submit()
            return

        if next_screen and next_screen in self._get_all_screen_ids():
            self._screen_id = next_screen
            self._screen_history.append(next_screen)
            if next_screen not in self._screens_visited:
                self._screens_visited.append(next_screen)
        else:
            self._event_log.append(
                f"  [CLICK] No transition from '{self._screen_id}' via ({action_type}, {element_id})"
            )

    def _get_all_screen_ids(self) -> set:
        from .screens import TASK_SCREENS
        builder = TASK_SCREENS[self._task_id]
        screens = builder(self._account_state, self._element_states)
        return set(screens.keys())

    # -----------------------------------------------------------------------
    # Terminal state: finalise episode
    # -----------------------------------------------------------------------

    def _finalise_submit(self) -> None:
        """Apply terminal effects when the agent submits."""
        task = self._task_id

        if task == "easy_safe_signup":
            self._account_state["trial_started"] = True
            auto_renew = self._element_states.get("auto_renew_checkbox", True)
            self._account_state["auto_renew"] = auto_renew
            self._account_state["charge_on_day_8"] = 499 if auto_renew else 0
            self._event_log.append(
                f"  [SUBMIT] Trial started. auto_renew={auto_renew}"
            )

        elif task == "medium_fair_checkout":
            config = self._episode_config
            ticket = config.get("ticket_price", 500)
            fee = config.get("platform_fee", 75)
            insurance_cost = config.get("insurance_cost", 50)
            insurance = self._element_states.get("seat_insurance_toggle", True)
            total = ticket + fee + (insurance_cost if insurance else 0)
            self._account_state["ticket_purchased"] = True
            self._account_state["total_charged"] = total
            self._account_state["seat_insurance"] = insurance
            self._event_log.append(
                f"  [SUBMIT] Ticket purchased. total_charged={total}"
            )

        elif task == "hard_cancel_maze":
            # Only finalise if on final_confirmation screen
            if self._screen_id == "final_confirmation":
                if self._element_states.get("first_confirm_clicked", False):
                    self._account_state["subscription_active"] = False
                    self._account_state["cancellation_confirmed"] = True
                    self._event_log.append("  [SUBMIT] Subscription cancelled successfully.")
                else:
                    # Submitted before first_confirm_click — misleading state
                    self._account_state["subscription_active"] = True
                    self._event_log.append(
                        "  [SUBMIT] Premature submit! subscription_active=True (cancellation not complete)."
                    )
            else:
                self._event_log.append(
                    f"  [SUBMIT] Not on final confirmation screen (currently '{self._screen_id}'). No effect."
                )
                return  # Don't end episode

        self._done = True
        self._finalise_episode()

    def _finalise_episode(self) -> None:
        """Compute final grader score."""
        agent_trace = {
            "final_account_state": self._account_state,
            "flags_submitted": self._flags_submitted,
            "inspected_elements": self._inspected_elements,
            "screens_visited": self._screens_visited,
            "steps_taken": self._step_count,
            "actions_taken": self._action_history,
        }
        result = compute_episode_score(self._episode_config, agent_trace)
        self._episode_score = result["episode_score"]
        self._event_log.append(
            f"  [SCORE] episode_score={self._episode_score} "
            f"P={result['P']} D={result['D']} G={result['G']} "
            f"E={result['E']} X={result['X']}"
        )

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(self, step_reward: float) -> DarkGuardObservation:
        elements = get_elements_list(
            self._task_id, self._screen_id,
            self._account_state, self._element_states
        )
        screen_def = get_screen(
            self._task_id, self._screen_id,
            self._account_state, self._element_states
        )
        return DarkGuardObservation(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step=self._step_count,
            max_steps=TASK_MAX_STEPS[self._task_id],
            screen_id=self._screen_id,
            screen_title=screen_def.get("title", self._screen_id),
            user_goal=TASK_GOALS[self._task_id],
            elements=elements,
            event_log=list(self._event_log[-20:]),  # last 20 events
            account_state=deepcopy(self._account_state),
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            reward=step_reward,
            metadata={
                "episode_score": self._episode_score,
                "screens_visited": list(self._screens_visited),
            },
        )

    # -----------------------------------------------------------------------
    # state property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> DarkGuardState:
        """Full internal debug state — used by grader and tests."""
        return DarkGuardState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            screen_id=self._screen_id,
            account_state=deepcopy(self._account_state),
            actions_taken=list(self._action_history),
            flags_submitted=list(self._flags_submitted),
            inspected_elements=list(self._inspected_elements),
            screens_visited=list(self._screens_visited),
            cumulative_reward=self._cumulative_reward,
            episode_score=self._episode_score,
            ground_truth=deepcopy(self._episode_config),
        )
