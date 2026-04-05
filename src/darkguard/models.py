"""
DarkGuard Pydantic Models.

Extends OpenEnv base types to define the typed API for the DarkGuard
consumer-protection RL environment.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# UI element — represents a single interactive widget on a screen
# ---------------------------------------------------------------------------

class UIElement(Observation):
    """A single interactable element on the current screen."""

    model_config = {"extra": "allow"}

    id: str = Field(..., description="Unique element identifier")
    type: Literal[
        "button", "checkbox", "toggle", "input",
        "text", "price_label", "fee_line", "timer",
        "link", "banner", "menu_item",
    ] = Field(..., description="Element type")
    label: str = Field(..., description="Visible label text shown to the agent")
    visible: bool = Field(default=True, description="Whether the element is visible")
    enabled: bool = Field(default=True, description="Whether the element is interactive")
    selected: bool = Field(default=False, description="Toggle/checkbox selected state")
    element_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hidden metadata revealed only via inspect() action",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DarkGuardAction(Action):
    """Action in the DarkGuard environment."""

    action_type: Literal[
        "click",    # advance flow / trigger transition
        "toggle",   # flip a checkbox or switch
        "type",     # fill a text field
        "inspect",  # reveal hidden element_metadata
        "go_back",  # return to previous screen
        "submit",   # commit current form state
        "flag",     # mark element as suspicious
    ] = Field(..., description="Type of action to perform")
    element_id: Optional[str] = Field(default=None, description="Target element ID")
    value: Optional[str] = Field(default=None, description="Text value for 'type' actions")
    note: Optional[str] = Field(default=None, description="Reason note for 'flag' actions")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class DarkGuardObservation(Observation):
    """Full observation returned by reset() and step()."""

    episode_id: str = Field(default="", description="Unique episode identifier")
    task_id: str = Field(default="", description="Task being run")
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=15, description="Maximum steps allowed")
    screen_id: str = Field(default="", description="Current screen identifier")
    screen_title: str = Field(default="", description="Human-readable screen title")
    user_goal: str = Field(default="", description="Plain-English goal the user wants to achieve")
    elements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Visible UI elements on the current screen",
    )
    event_log: List[str] = Field(
        default_factory=list,
        description="History of actions and events in this episode",
    )
    account_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current account/outcome state (e.g. charged, subscribed)",
    )
    step_reward: float = Field(default=0.0, description="Reward received on the previous step")
    cumulative_reward: float = Field(default=0.0, description="Running reward total")
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "click", "toggle", "type", "inspect", "go_back", "submit", "flag"
        ],
        description="Action types valid on this screen",
    )


# ---------------------------------------------------------------------------
# State  (internal debug state — returned by state() endpoint)
# ---------------------------------------------------------------------------

class DarkGuardState(State):
    """Full internal debug state returned by state() endpoint."""

    task_id: str = Field(default="", description="Active task ID")
    screen_id: str = Field(default="", description="Current screen ID")
    account_state: Dict[str, Any] = Field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    flags_submitted: List[Dict[str, Any]] = Field(default_factory=list)
    inspected_elements: List[str] = Field(default_factory=list)
    screens_visited: List[str] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    episode_score: Optional[float] = Field(default=None)
    ground_truth: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ground-truth trap config (hidden from agent, exposed here for grader/tests)",
    )
