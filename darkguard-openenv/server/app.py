"""OpenEnv-compatible FastAPI server exposing DarkGuard."""

from __future__ import annotations

import threading

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment

from darkguard_openenv.environment import DarkGuardEnvironment
from darkguard_openenv.models import (
    DarkGuardOpenEnvAction,
    DarkGuardOpenEnvObservation,
    DarkGuardOpenEnvState,
)


class DarkGuardOpenEnvAdapter(Environment[DarkGuardOpenEnvAction, DarkGuardOpenEnvObservation, DarkGuardOpenEnvState]):
    """Adapter wrapping dict-based core env into OpenEnv Environment interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _sessions: dict[str, DarkGuardEnvironment] = {}
    _last_env = DarkGuardEnvironment()
    _lock = threading.RLock()

    def __init__(self) -> None:
        super().__init__()
        self._env = DarkGuardEnvironment()

    @classmethod
    def _session_env(cls, episode_id: str | None) -> DarkGuardEnvironment:
        if episode_id and episode_id in cls._sessions:
            return cls._sessions[episode_id]
        return cls._last_env

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: object) -> DarkGuardOpenEnvObservation:
        payload = dict(kwargs)
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        with self._lock:
            obs = self._env.reset(**payload)
            resolved_id = obs.get("episode_id")
            if resolved_id:
                self._sessions[str(resolved_id)] = self._env
            self._last_env = self._env
        return DarkGuardOpenEnvObservation(**obs)

    def step(
        self,
        action: DarkGuardOpenEnvAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> DarkGuardOpenEnvObservation:
        _ = timeout_s
        action_payload = action.model_dump(exclude_none=True)
        # OpenEnv Action metadata can include auxiliary fields unknown to strict validator.
        metadata = action_payload.pop("metadata", {}) if isinstance(action_payload.get("metadata"), dict) else {}
        episode_id = (
            kwargs.get("episode_id")
            or action_payload.get("episode_id")
            or metadata.get("episode_id")
        )
        env = self._session_env(str(episode_id)) if episode_id else self._last_env
        try:
            obs = env.step(action_payload)
        except RuntimeError:
            # Defensive fallback for stateless request paths.
            env.reset()
            obs = env.step(action_payload)
        with self._lock:
            resolved_id = obs.get("episode_id")
            if resolved_id:
                self._sessions[str(resolved_id)] = env
            self._last_env = env
        return DarkGuardOpenEnvObservation(**obs)

    @property
    def state(self) -> DarkGuardOpenEnvState:
        with self._lock:
            return DarkGuardOpenEnvState(**self._last_env.state())


app = create_app(
    env=DarkGuardOpenEnvAdapter,
    action_cls=DarkGuardOpenEnvAction,
    observation_cls=DarkGuardOpenEnvObservation,
    env_name="darkguard-openenv",
)
