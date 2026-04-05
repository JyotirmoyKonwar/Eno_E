"""
FastAPI application for the DarkGuard Environment.

Endpoints:
    POST /reset  — Reset environment (accepts optional task_id in body)
    POST /step   — Execute an action
    GET  /state  — Full internal debug state
    GET  /schema — Action/observation schemas
    WS   /ws     — WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..src.darkguard.models import DarkGuardAction, DarkGuardObservation
    from .darkguard_environment import DarkGuardEnvironment
except (ModuleNotFoundError, ImportError):
    from src.darkguard.models import DarkGuardAction, DarkGuardObservation
    from server.darkguard_environment import DarkGuardEnvironment


app = create_app(
    DarkGuardEnvironment,
    DarkGuardAction,
    DarkGuardObservation,
    env_name="DarkGuard",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run or direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Call main() — port override passed separately
    _port = args.port
    main() if _port == 8000 else main(port=_port)
