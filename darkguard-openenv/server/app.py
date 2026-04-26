"""OpenEnv-compatible FastAPI server exposing DarkGuard."""

from __future__ import annotations

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

    def __init__(self) -> None:
        super().__init__()
        self._env = DarkGuardEnvironment()

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: object) -> DarkGuardOpenEnvObservation:
        payload = dict(kwargs)
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
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
        try:
            obs = self._env.step(action_payload)
        except RuntimeError:
            # Defensive fallback for stateless request paths.
            self._env.reset()
            obs = self._env.step(action_payload)
        return DarkGuardOpenEnvObservation(**obs)

    @property
    def state(self) -> DarkGuardOpenEnvState:
        return DarkGuardOpenEnvState(**self._env.state())


app = create_app(
    env=DarkGuardOpenEnvAdapter,
    action_cls=DarkGuardOpenEnvAction,
    observation_cls=DarkGuardOpenEnvObservation,
    env_name="darkguard-openenv",
)
