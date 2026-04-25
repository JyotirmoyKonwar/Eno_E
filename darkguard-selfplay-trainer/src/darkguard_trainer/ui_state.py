"""Thread-safe runtime state for Gradio UI."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeState:
    running: bool = False
    stop_requested: bool = False
    current_round: int = 0
    active_phase: str = "idle"
    logs: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    latest_eval: dict[str, Any] = field(default_factory=dict)


class StateHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RuntimeState()

    def snapshot(self) -> RuntimeState:
        with self._lock:
            s = self._state
            return RuntimeState(
                running=s.running,
                stop_requested=s.stop_requested,
                current_round=s.current_round,
                active_phase=s.active_phase,
                logs=list(s.logs[-500:]),
                metrics=list(s.metrics),
                artifacts=dict(s.artifacts),
                latest_eval=dict(s.latest_eval),
            )

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)

    def append_log(self, line: str) -> None:
        with self._lock:
            self._state.logs.append(line)

    def append_metric(self, metric: dict[str, Any]) -> None:
        with self._lock:
            self._state.metrics.append(metric)
