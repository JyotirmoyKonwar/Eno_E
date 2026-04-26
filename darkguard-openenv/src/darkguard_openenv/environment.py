"""Core DarkGuard OpenEnv environment implementation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .models import ActionType, DarkGuardAction, DarkGuardObservation, DarkGuardState, UIElement
from .rewards import RewardAccumulator, RewardContext, clip_reward
from .screens import ScreenDefinition, TaskDefinition, builtin_tasks
from .utils import new_episode_id, norm_text
from .validators import validate_action_payload, validate_custom_episode


DEFAULT_MAX_STEPS = 25
ALLOWED_ACTIONS = [a.value for a in ActionType if a != ActionType.INVALID]


@dataclass(slots=True)
class EpisodeRuntime:
    task: TaskDefinition
    screen_id: str
    episode_id: str
    rng: random.Random
    max_steps: int
    subtlety: float
    done: bool = False
    cumulative_reward: float = 0.0
    step_count: int = 0
    messages: list[str] = field(default_factory=list)
    visited: dict[tuple[str, str], int] = field(default_factory=dict)
    inspected_targets: set[str] = field(default_factory=set)
    flagged_targets: set[str] = field(default_factory=set)
    last_action_result: str = "Environment initialized."
    reward_acc: RewardAccumulator = field(default_factory=RewardAccumulator)
    outcome_summary: str = "in_progress"


class DarkGuardEnvironment:
    """Gym-style environment with reset/step/state contract."""

    def __init__(self) -> None:
        self._builtin = builtin_tasks()
        self._episode: EpisodeRuntime | None = None

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        task_id = kwargs.get("task_id")
        seed = kwargs.get("seed")
        max_steps = int(kwargs.get("max_steps", DEFAULT_MAX_STEPS))
        difficulty = kwargs.get("difficulty", "medium")
        subtlety = float(kwargs.get("subtlety", 0.5))
        episode_config = kwargs.get("episode_config")

        rng = random.Random(seed)
        if max_steps < 1:
            max_steps = DEFAULT_MAX_STEPS

        task = self._resolve_task(task_id=task_id, difficulty=difficulty, episode_config=episode_config, rng=rng)
        self._episode = EpisodeRuntime(
            task=task,
            screen_id=task.start_screen_id,
            episode_id=new_episode_id(),
            rng=rng,
            max_steps=max_steps,
            subtlety=max(0.0, min(1.0, subtlety)),
            messages=[f"Task loaded: {task.task_id}"],
        )
        return self._observation(0.0, {"total": 0.0})

    def state(self) -> dict[str, Any]:
        ep = self._require_episode()
        return DarkGuardState(
            episode_id=ep.episode_id,
            task_id=ep.task.task_id,
            screen_id=ep.screen_id,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            cumulative_reward=ep.cumulative_reward,
            done=ep.done,
            outcome_summary=ep.outcome_summary,
            messages=ep.messages[-8:],
            reward_totals=ep.reward_acc.totals.as_dict(),
        ).to_dict()

    def step(self, action: dict[str, Any] | str) -> dict[str, Any]:
        ep = self._require_episode()
        if ep.done:
            ep.last_action_result = "Episode already done. Call reset()."
            rb = ep.reward_acc.totals.as_dict()
            return self._observation(0.0, rb)

        parsed_action = self.parse_action(action)
        current_screen = ep.task.screens[ep.screen_id]
        reward_ctx = RewardContext(step_count=ep.step_count + 1, max_steps=ep.max_steps)
        action_key = f"{parsed_action.action_type.value}:{parsed_action.target_id or '-'}"
        state_action_key = (ep.screen_id, action_key)
        ep.visited[state_action_key] = ep.visited.get(state_action_key, 0) + 1
        reward_ctx.repeated_state = ep.visited[state_action_key] > 2

        self._apply_action(parsed_action, ep, current_screen, reward_ctx)

        ep.step_count += 1
        if ep.step_count >= ep.max_steps and not ep.done:
            ep.done = True
            ep.outcome_summary = "max_steps_reached"
            ep.last_action_result = "Max steps reached."

        if ep.done:
            reward_ctx.terminal_safe = ep.screen_id in ep.task.safe_terminal_ids
            reward_ctx.terminal_harmful = ep.screen_id in ep.task.harmful_terminal_ids
            if reward_ctx.terminal_safe:
                ep.outcome_summary = "safe_completion"
            elif reward_ctx.terminal_harmful:
                ep.outcome_summary = "harmful_completion"

        rb = ep.reward_acc.update(parsed_action, reward_ctx)
        reward = clip_reward(rb.total())
        ep.cumulative_reward += reward
        return self._observation(reward, rb.as_dict())

    def parse_action(self, raw: dict[str, Any] | str) -> DarkGuardAction:
        if isinstance(raw, dict):
            action, error = validate_action_payload(raw)
            if error:
                action.parser_error = "payload_validation_error"
            return action

        text = str(raw or "").strip()
        if not text:
            return DarkGuardAction(action_type=ActionType.INVALID, raw_text=text, parser_error="empty_action")

        try:
            maybe_json = json.loads(text)
            if isinstance(maybe_json, dict):
                action, error = validate_action_payload(maybe_json)
                if error:
                    action.parser_error = "json_validation_error"
                action.raw_text = text
                return action
        except json.JSONDecodeError:
            pass

        pattern = re.compile(
            r"ACTION:\s*(?P<action>[a-z_]+)"
            r"(?:\s*\|\s*TARGET:\s*(?P<target>[a-zA-Z0-9_\-]+))?"
            r"(?:\s*\|\s*CATEGORY:\s*(?P<category>[a-zA-Z0-9_\-]+))?"
            r"(?:\s*\|\s*NOTES:\s*(?P<notes>.*))?$",
            re.IGNORECASE,
        )
        match = pattern.match(text)
        if match:
            action_name = norm_text(match.group("action"))
            if action_name in ALLOWED_ACTIONS:
                return DarkGuardAction(
                    action_type=ActionType(action_name),
                    target_id=match.group("target"),
                    flag_category=match.group("category"),
                    notes=match.group("notes"),
                    raw_text=text,
                )

        return DarkGuardAction(action_type=ActionType.INVALID, raw_text=text, parser_error="parse_failure")

    def _resolve_task(
        self,
        *,
        task_id: str | None,
        difficulty: str,
        episode_config: dict[str, Any] | None,
        rng: random.Random,
    ) -> TaskDefinition:
        try:
            if (task_id == "custom_episode" or episode_config) and episode_config:
                validated = validate_custom_episode(episode_config)
                return self._task_from_config(validated.model_dump())
        except Exception as e:
            # Safety net: if designer generates junk, fallback to a stable builtin
            pass

        if task_id and task_id in self._builtin:
            return self._builtin[task_id]

        options = ["easy_safe_signup", "medium_fair_checkout", "hard_cancel_maze"]
        return self._builtin[rng.choice(options)]

    def _task_from_config(self, cfg: dict[str, Any]) -> TaskDefinition:
        screens: dict[str, ScreenDefinition] = {}
        for s in cfg["screens"]:
            elements = [
                UIElement(
                    id=e["id"],
                    type=e["type"],
                    text=e["text"],
                    checked=e.get("checked", False),
                    enabled=e.get("enabled", True),
                    prominence=e.get("prominence", 1),
                    metadata=e.get("metadata", {}),
                )
                for e in s["elements"]
            ]
            screens[s["screen_id"]] = ScreenDefinition(
                screen_id=s["screen_id"],
                description=s["description"],
                elements=elements,
                transitions=s.get("transitions", {}),
                terminal=bool(s.get("terminal", False)),
            )
        return TaskDefinition(
            task_id=cfg["task_id"],
            instruction=cfg["instruction"],
            start_screen_id=cfg["start_screen_id"],
            screens=screens,
            safe_terminal_ids=set(cfg["safe_terminal_ids"]),
            harmful_terminal_ids=set(cfg["harmful_terminal_ids"]),
            trap_map=cfg.get("trap_map", {}),
        )

    def _apply_action(
        self,
        action: DarkGuardAction,
        ep: EpisodeRuntime,
        current_screen: ScreenDefinition,
        reward_ctx: RewardContext,
    ) -> None:
        if action.action_type == ActionType.INVALID:
            reward_ctx.was_invalid = True
            ep.last_action_result = f"Invalid action format ({action.parser_error or 'unknown'})."
            ep.messages.append(ep.last_action_result)
            return

        target = action.target_id
        elements = {e.id: e for e in current_screen.elements}

        if action.action_type == ActionType.INSPECT:
            if not target or target not in elements:
                reward_ctx.was_invalid = True
                ep.last_action_result = "inspect requires a visible target_id."
            elif target in ep.inspected_targets:
                reward_ctx.was_invalid = True
                ep.last_action_result = f"{target} already inspected in this episode."
            else:
                reward_ctx.inspected_new_target = target not in ep.inspected_targets
                ep.inspected_targets.add(target)
                ep.last_action_result = f"Inspected {target}: {elements[target].text}"

        elif action.action_type == ActionType.FLAG:
            if not target:
                reward_ctx.was_invalid = True
                ep.last_action_result = "flag requires target_id."
            else:
                is_true_trap = target in ep.task.trap_map
                reward_ctx.correct_flag = is_true_trap and target not in ep.flagged_targets
                reward_ctx.false_flag = not is_true_trap
                ep.flagged_targets.add(target)
                if is_true_trap and action.flag_category:
                    trap_cat = norm_text(ep.task.trap_map[target].get("category"))
                    reward_ctx.evidence_match = trap_cat in norm_text(action.flag_category)
                if is_true_trap and action.notes:
                    reward_ctx.evidence_match = reward_ctx.evidence_match or target in norm_text(action.notes)
                ep.last_action_result = f"Flag submitted for {target}."

        elif action.action_type in {ActionType.CLICK, ActionType.TOGGLE}:
            if not target or target not in elements:
                reward_ctx.was_invalid = True
                ep.last_action_result = f"{action.action_type.value} target missing or not visible."
            else:
                if action.action_type == ActionType.TOGGLE and elements[target].type in {"checkbox", "toggle"}:
                    elements[target].checked = not elements[target].checked
                    ep.last_action_result = f"Toggled {target} to {elements[target].checked}."
                elif target in current_screen.transitions:
                    ep.screen_id = current_screen.transitions[target]
                    ep.last_action_result = f"Navigated to {ep.screen_id}."
                    if ep.task.screens[ep.screen_id].terminal:
                        ep.done = True
                else:
                    ep.last_action_result = f"Clicked {target}. No visible transition."

        elif action.action_type == ActionType.GO_BACK:
            if "back" in current_screen.transitions:
                ep.screen_id = current_screen.transitions["back"]
                ep.last_action_result = f"Navigated back to {ep.screen_id}."
            else:
                reward_ctx.was_invalid = True
                ep.last_action_result = "go_back unavailable on this screen."

        elif action.action_type == ActionType.SUBMIT:
            if current_screen.terminal:
                ep.done = True
                ep.last_action_result = "Submitted terminal decision."
            else:
                # Anti-hacking: no shortcut submission from non-terminal screens.
                reward_ctx.was_invalid = True
                ep.last_action_result = "submit only allowed on terminal state."

        ep.messages.append(ep.last_action_result)

    def _observation(self, reward: float, reward_breakdown: dict[str, float]) -> dict[str, Any]:
        ep = self._require_episode()
        screen = ep.task.screens[ep.screen_id]
        elements = list(screen.elements)
        allowed_actions: list[str] = []
        clickable = [e for e in elements if e.enabled and e.type in {"button", "link", "checkbox", "toggle"}]
        togglable = [e for e in elements if e.enabled and e.type in {"checkbox", "toggle"}]
        inspectable = [e for e in elements if e.id not in ep.inspected_targets]
        flaggable = [e for e in elements if e.id not in ep.flagged_targets]
        if inspectable:
            allowed_actions.append(ActionType.INSPECT.value)
        if clickable:
            allowed_actions.append(ActionType.CLICK.value)
        if togglable:
            allowed_actions.append(ActionType.TOGGLE.value)
        if flaggable:
            allowed_actions.append(ActionType.FLAG.value)
        if "back" in screen.transitions:
            allowed_actions.append(ActionType.GO_BACK.value)
        if screen.terminal:
            allowed_actions.append(ActionType.SUBMIT.value)
        obs = DarkGuardObservation(
            episode_id=ep.episode_id,
            task_id=ep.task.task_id,
            screen_id=ep.screen_id,
            instruction=ep.task.instruction,
            visible_summary=screen.description,
            elements=screen.elements,
            allowed_actions=allowed_actions,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            reward=reward,
            cumulative_reward=ep.cumulative_reward,
            done=ep.done,
            last_action_result=ep.last_action_result,
            messages=ep.messages[-8:],
            reward_breakdown=reward_breakdown,
        )
        return obs.to_dict()

    def _require_episode(self) -> EpisodeRuntime:
        if not self._episode:
            raise RuntimeError("Episode not initialized. Call reset first.")
        return self._episode
