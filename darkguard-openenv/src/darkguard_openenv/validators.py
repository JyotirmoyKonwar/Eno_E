"""Validation helpers for actions and custom episodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .models import ActionType, DarkGuardAction


class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str
    target_id: str | None = None
    flag_category: str | None = None
    notes: str | None = None

    @field_validator("action_type")
    @classmethod
    def _validate_action_type(cls, value: str) -> str:
        if value not in {a.value for a in ActionType if a != ActionType.INVALID}:
            raise ValueError("Unsupported action_type")
        return value


class ElementPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    text: str
    checked: bool = False
    enabled: bool = True
    prominence: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScreenPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    screen_id: str
    description: str
    elements: list[ElementPayload]
    transitions: dict[str, str] = Field(default_factory=dict)
    terminal: bool = False


class EpisodeConfigPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = "custom_episode"
    instruction: str
    start_screen_id: str
    screens: list[ScreenPayload]
    safe_terminal_ids: list[str]
    harmful_terminal_ids: list[str]
    trap_map: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("safe_terminal_ids", "harmful_terminal_ids")
    @classmethod
    def _validate_terminal_sets(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("terminal set cannot be empty")
        return value


def validate_action_payload(payload: dict[str, Any]) -> tuple[DarkGuardAction, str | None]:
    try:
        model = ActionPayload.model_validate(payload)
    except ValidationError as exc:
        return DarkGuardAction(action_type=ActionType.INVALID, parser_error="invalid action payload"), str(exc)
    return (
        DarkGuardAction(
            action_type=ActionType(model.action_type),
            target_id=model.target_id,
            flag_category=model.flag_category,
            notes=model.notes,
        ),
        None,
    )


def validate_custom_episode(payload: dict[str, Any]) -> EpisodeConfigPayload:
    return EpisodeConfigPayload.model_validate(payload)
