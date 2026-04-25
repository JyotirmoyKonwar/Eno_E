"""Parser helpers for action-like text outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_action_text(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {"action_type": "inspect", "target_id": None, "notes": "empty_parse"}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    pattern = re.compile(
        r"ACTION:\s*(?P<action>[a-z_]+)(?:\s*\|\s*TARGET:\s*(?P<target>[a-zA-Z0-9_\-]+))?(?:\s*\|\s*NOTES:\s*(?P<notes>.*))?$",
        re.IGNORECASE,
    )
    match = pattern.match(text)
    if not match:
        return {"action_type": "inspect", "target_id": None, "notes": f"unparsed:{text[:40]}"}
    return {
        "action_type": match.group("action").lower(),
        "target_id": match.group("target"),
        "notes": match.group("notes"),
    }
