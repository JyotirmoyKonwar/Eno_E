"""
DarkGuard Environment Server Adapter.

Thin wrapper that plugs DarkGuardEnv into the OpenEnv HTTP server.
"""

from typing import Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..src.darkguard.env import DarkGuardEnv
    from ..src.darkguard.models import DarkGuardAction, DarkGuardObservation, DarkGuardState
except ImportError:
    from src.darkguard.env import DarkGuardEnv
    from src.darkguard.models import DarkGuardAction, DarkGuardObservation, DarkGuardState


class DarkGuardEnvironment(Environment):
    """
    OpenEnv server adapter for DarkGuardEnv.

    Delegates all calls to DarkGuardEnv while satisfying the
    Environment[DarkGuardAction, DarkGuardObservation, DarkGuardState] interface.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._env = DarkGuardEnv()

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DarkGuardObservation:
        return self._env.reset(task_id=task_id, seed=seed, episode_id=episode_id)

    def step(
        self,
        action: DarkGuardAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DarkGuardObservation:
        return self._env.step(action)

    @property
    def state(self) -> DarkGuardState:
        return self._env.state
