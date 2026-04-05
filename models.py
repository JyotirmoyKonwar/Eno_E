"""
DarkGuard — Root-level model re-exports.

Exposes DarkGuard typed models at the package root for backward
compatibility with existing imports (e.g. from models import ...).
"""

try:
    from src.darkguard.models import (
        DarkGuardAction,
        DarkGuardObservation,
        DarkGuardState,
        UIElement,
    )
except ImportError:
    from darkguard.models import (  # type: ignore[no-redef]
        DarkGuardAction,
        DarkGuardObservation,
        DarkGuardState,
        UIElement,
    )

__all__ = [
    "DarkGuardAction",
    "DarkGuardObservation",
    "DarkGuardState",
    "UIElement",
]
