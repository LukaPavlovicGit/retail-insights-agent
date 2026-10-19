from __future__ import annotations

from pathlib import Path

DEFAULT_PERSONA = "You are a plain-spoken retail analytics assistant. Lead with the answer, then the evidence."


def load_persona(path: Path) -> str:
    p = Path(path)
    if not p.exists():
        return DEFAULT_PERSONA
    text = p.read_text(encoding="utf-8").strip()
    return text or DEFAULT_PERSONA
