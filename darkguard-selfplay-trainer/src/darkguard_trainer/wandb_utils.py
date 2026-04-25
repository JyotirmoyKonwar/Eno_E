"""Weights & Biases integration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WandbSession:
    enabled: bool
    run_url: str = ""
    _run: Any | None = None

    def log(self, payload: dict[str, Any], step: int) -> None:
        if self.enabled and self._run is not None:
            self._run.log(payload, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()


def init_wandb(token: str | None, enabled: bool, project: str, config: dict[str, Any]) -> tuple[WandbSession, str]:
    if not enabled:
        return WandbSession(enabled=False), "W&B disabled."
    try:
        import wandb

        resolved_token = (token or os.getenv("WANDB_API_KEY") or "").strip() or None
        if resolved_token:
            wandb.login(key=resolved_token, relogin=True)
        run = wandb.init(
            project=project,
            config=config,
            tags=["darkguard", "selfplay", "grpo-demo"],
            settings=wandb.Settings(start_method="thread"),
        )
        session = WandbSession(enabled=True, run_url=run.url if run else "", _run=run)
        return session, "W&B connected."
    except Exception as exc:  # pragma: no cover - runtime integration path
        return WandbSession(enabled=False), f"W&B unavailable: {exc}"
