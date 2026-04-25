"""DarkGuard self-play trainer package."""

__all__ = ["build_app"]


def build_app():
    """Lazy import so tests and tooling can import trainer modules without Gradio."""
    from .gradio_app import build_app as _build_app

    return _build_app()
