"""Load the assistant's persona text (voice and tone) that opens every system prompt, and the business definitions.

Read by `Deps.persona` from `data/persona.md` and `Deps.definitions` from `data/definitions.md`, so both can be
edited without code changes.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PERSONA = "You are a plain-spoken retail analytics assistant. Lead with the answer, then the evidence."


DEFAULT_DEFINITIONS = """# Business definitions (editable; read on every answer)

- Churn: a customer who made no purchase in the last 90 days. Churn rate for a period is churned customers over customers seen.
- Revenue: sum of order_items.sale_price for items with status Complete, Shipped or Processing. Cancelled and Returned are excluded.
- Active customer: at least one purchase in the period.
- New customer: first purchase falls in the period.

If the user specifies a different window or statuses in their question, use theirs for that answer and say so."""


def load_definitions(path: Path) -> str:
    """Return the trimmed contents of `path` (`data/definitions.md`), or `DEFAULT_DEFINITIONS` if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return DEFAULT_DEFINITIONS
    text = p.read_text(encoding="utf-8").strip()
    return text or DEFAULT_DEFINITIONS


def load_persona(path: Path) -> str:
    """Return the trimmed contents of `path`, or `DEFAULT_PERSONA` if the file is missing or empty."""
    p = Path(path)
    if not p.exists():
        return DEFAULT_PERSONA
    text = p.read_text(encoding="utf-8").strip()
    return text or DEFAULT_PERSONA
