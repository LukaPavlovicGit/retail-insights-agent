"""Quality gate for persona changes.

The persona (`data/persona.md`) is the one prompt file that non-developers edit. The client requires that a
persona change passes an automated quality gate, with no manual approval step, so this gate is all that stands
between an edit and production. `lint_persona` is the offline check (length and forbidden topics),
`live_gate` runs the eval set with the candidate persona, and `publish_persona` swaps the file in while keeping the
previous version. The CLI (`ria persona check|publish`) runs the checks before it publishes.
"""
from __future__ import annotations

import dataclasses
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

MAX_PERSONA_CHARS = 4000
JUDGE_THRESHOLD = 4.0

FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore (?:(?:all|previous|the) )+instructions", "tries to override instructions"),
    (r"\bbrands?\b", "mentions brands (scope is not the persona's job)"),
    (r"\bscope\b", "mentions scope"),
    (r"\bemail\b", "mentions email (personal data is handled by the guard)"),
    (r"\bsql\b", "mentions SQL"),
    (r"\bselect\b", "mentions SELECT"),
    (r"\bdelete\b", "mentions delete"),
    (r"\bpii\b", "mentions PII"),
    # Naming churn or revenue as a topic is fine ("round revenue to the nearest thousand"); redefining them is not.
    (r"\b(?:churn|revenue)\b[^.\n]{0,60}\b(?:is|means|defined?|counts?|includes?|excludes?|window|days?|status(?:es)?)\b",
     "redefines churn or revenue (business definitions live in org_defaults.yaml)"),
    (r"\b(?:is|means|defined?|counts?|includes?|excludes?|window|days?|status(?:es)?)\b[^.\n]{0,60}\b(?:churn|revenue)\b",
     "redefines churn or revenue (business definitions live in org_defaults.yaml)"),
    (r"\bminimum group\b", "mentions the minimum group size (org_defaults.yaml)"),
)
_COMPILED = tuple((re.compile(p, re.IGNORECASE), reason) for p, reason in FORBIDDEN_PATTERNS)


def lint_persona(text: str) -> list[str]:
    """Return the reasons `text` is rejected; an empty list means it passes. Order: empty or length first, then
    by line number, then by pattern order."""
    if not text.strip():
        return ["persona is empty"]
    reasons = [f"persona is {len(text)} characters, limit is {MAX_PERSONA_CHARS}"] if len(text) > MAX_PERSONA_CHARS else []
    return reasons + [
        f"line {n}: {reason}"
        for n, line in enumerate(text.splitlines(), 1)
        for rx, reason in _COMPILED
        if rx.search(line)
    ]


async def live_gate(settings, conn, cases, candidate: Path) -> tuple[bool, str]:
    """Run the eval `cases` with `candidate` as the persona. Passes when no case has a deterministic failure and
    the mean judge score is at least `JUDGE_THRESHOLD`. Returns (passed, one-line summary). Needs credentials."""
    from ria.evals.runner import run_evals

    rows = await run_evals(dataclasses.replace(settings, persona_override=candidate), conn, cases)
    failed = sum(1 for r in rows if r["failures"])
    scores = [r["score"] for r in rows if r["score"] is not None]
    mean = sum(scores) / len(scores) if scores else None
    summary = f"{len(rows)} cases, {failed} deterministic failures, judge mean {'n/a' if mean is None else f'{mean:.1f}'}"
    return failed == 0 and mean is not None and mean >= JUDGE_THRESHOLD, summary


def publish_persona(candidate: Path, target: Path, versions_dir: Path, now: datetime | None = None) -> Path | None:
    """Copy the current `target` to `versions_dir/<UTC YYYYMMDDTHHMMSSZ>.md`, then write the candidate text to
    `target`. Returns the saved version, or None if there was no previous persona. Does not lint."""
    version = None
    if target.exists():
        stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        versions_dir.mkdir(parents=True, exist_ok=True)
        version = versions_dir / f"{stamp}.md"
        shutil.copyfile(target, version)
    target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
    return version
