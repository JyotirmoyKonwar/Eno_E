"""HF Spaces entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

# Gradio SDK Spaces do not automatically add /app/src to PYTHONPATH.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from darkguard_trainer.gradio_app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
