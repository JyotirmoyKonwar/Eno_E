"""OpenEnv-compatible FastAPI server exposing DarkGuard."""

from __future__ import annotations

import threading

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment

from DarkVader_openenv.environment import DarkGuardEnvironment
from DarkVader_openenv.models import (
    DarkGuardOpenEnvAction,
    DarkGuardOpenEnvObservation,
    DarkGuardOpenEnvState,
)


class DarkGuardOpenEnvAdapter(Environment[DarkGuardOpenEnvAction, DarkGuardOpenEnvObservation, DarkGuardOpenEnvState]):
    """Adapter wrapping dict-based core env into OpenEnv Environment interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _shared_env = DarkGuardEnvironment()
    _shared_lock = threading.RLock()

    def __init__(self) -> None:
        super().__init__()
        self._env = self._shared_env

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: object) -> DarkGuardOpenEnvObservation:
        payload = dict(kwargs)
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        with self._shared_lock:
            obs = self._env.reset(**payload)
        return DarkGuardOpenEnvObservation(**obs)

    def step(
        self,
        action: DarkGuardOpenEnvAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> DarkGuardOpenEnvObservation:
        _ = timeout_s, kwargs
        action_payload = action.model_dump(exclude_none=True)
        # OpenEnv may include auxiliary metadata on action objects; DarkGuard
        # validator is strict and should only see canonical action fields.
        action_payload.pop("metadata", None)
        with self._shared_lock:
            try:
                obs = self._env.step(action_payload)
            except RuntimeError:
                # Defensive fallback for stateless request paths.
                self._env.reset()
                obs = self._env.step(action_payload)
        return DarkGuardOpenEnvObservation(**obs)

    @property
    def state(self) -> DarkGuardOpenEnvState:
        with self._shared_lock:
            return DarkGuardOpenEnvState(**self._env.state())


app = create_app(
    env=DarkGuardOpenEnvAdapter,
    action_cls=DarkGuardOpenEnvAction,
    observation_cls=DarkGuardOpenEnvObservation,
    env_name="DarkVader-openenv",
)
