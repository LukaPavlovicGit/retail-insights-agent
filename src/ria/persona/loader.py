"""Load the assistant's persona text (voice and tone) that opens every system prompt.

Read by `Deps.persona` from `data/persona.md`, so the voice can be edited without code changes.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PERSONA = "You are a plain-spoken retail analytics assistant. Lead with the answer, then the evidence."


def load_persona(path: Path) -> str:
    """Return the trimmed contents of `path`, or `DEFAULT_PERSONA` if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return DEFAULT_PERSONA
    text = p.read_text(encoding="utf-8").strip()
    return text or DEFAULT_PERSONA
