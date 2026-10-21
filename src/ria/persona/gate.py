"""Quality gate for persona changes.

The persona (`data/persona.md`) is the one prompt file that non-developers edit. The client requires that a
persona change passes an automated quality gate, with no manual approval step, so this gate is all that stands
between an edit and production. `lint_persona` is the offline check (length and forbidden topics),
`live_gate` runs the eval set with the candidate persona, and `publish_persona` swaps the file in while keeping the
previous version. The CLI (`ria persona check|publish`) runs the checks before it publishes.
"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

MAX_PERSONA_CHARS = 4000
JUDGE_THRESHOLD = 4.0

# Matched against each line.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:ignore|disregard|forget)\b[^.\n]{0,30}\b(?:instructions|rules)\b", "tries to override instructions"),
    (r"\b(?:answer|reply|respond|write|speak)\b[^.\n]{0,20}\bin (?!english\b)[a-z]+", "changes the output language (English only)"),
    (r"\b(?:output|answer|response|reply)\s+language\b|\blanguage\b[^.\n]{0,20}\b(?:spanish|french|german|serbian|italian|portuguese|chinese|japanese|arabic|russian|hindi)\b",
     "mentions language (English only)"),
    (r"\bbrands?\b", "mentions brands (scope is not the persona's job)"),
    (r"\bscope\b", "mentions scope"),
    (r"\be-?mails?\b", "mentions email (personal data is handled by the guard)"),
    (r"\bsql\b", "mentions SQL"),
    (r"\bselect\b", "mentions SELECT"),
    (r"\bdelete\b", "mentions delete"),
    (r"\bpii\b", "mentions PII"),
    (r"\bminimum group\b", "mentions the minimum group size (org_defaults.yaml)"),
)
# Naming churn or revenue as a topic is fine ("round revenue to the nearest thousand"); redefining them is not.
# Matched against units (see `_units`), so a heading plus its bullet is checked together.
_DEFINING = r"(?:is|means|defined?|counts?|includes?|excludes?|window|days?|status(?:es)?|complete\w*|shipped|cancell?ed|returned)"
_METRIC = r"(?:churn\w*|revenues?)"
_METRIC_REASON = "may redefine churn or revenue (business definitions live in org_defaults.yaml); reword if it only names the metric"
_METRIC_RX = (
    re.compile(rf"\b{_METRIC}\b[^.]{{0,80}}\b{_DEFINING}\b", re.IGNORECASE),
    re.compile(rf"\b{_DEFINING}\b[^.]{{0,80}}\b{_METRIC}\b", re.IGNORECASE),
)
_COMPILED = tuple((re.compile(p, re.IGNORECASE), reason) for p, reason in FORBIDDEN_PATTERNS)
_BULLET = re.compile(r"\s*[-*] ")


def _units(text: str):
    """Yield (first line number, unit text). A unit is a paragraph (split on blank lines) cut at each bullet, with a
    heading kept together with what follows it."""
    unit: list[str] = []
    start = 0
    for n, line in enumerate(text.splitlines() + [""], 1):
        if not line.strip():
            if unit:
                yield start, "\n".join(unit)
            unit = []
            continue
        starts_item = _BULLET.match(line) or line.lstrip().startswith("#")
        if unit and starts_item and not all(u.lstrip().startswith("#") for u in unit):
            yield start, "\n".join(unit)
            unit = []
        if not unit:
            start = n
        unit.append(line)


def lint_persona(text: str) -> list[str]:
    """Return the reasons `text` is rejected; an empty list means it passes. The text is NFKC-normalised and
    invisible format characters are removed first. Order: empty or length first, then by line number, then by
    pattern order (churn/revenue last)."""
    text = "".join(ch for ch in unicodedata.normalize("NFKC", text) if unicodedata.category(ch) != "Cf")
    if not text.strip():
        return ["persona is empty"]
    reasons = [f"persona is {len(text)} characters, limit is {MAX_PERSONA_CHARS}"] if len(text) > MAX_PERSONA_CHARS else []
    hits = {
        (n, i): reason
        for n, line in enumerate(text.splitlines(), 1)
        for i, (rx, reason) in enumerate(_COMPILED)
        if rx.search(line)
    }
    hits.update({(n, len(_COMPILED)): _METRIC_REASON for n, unit in _units(text) if any(rx.search(unit) for rx in _METRIC_RX)})
    return reasons + [f"line {n}: {reason}" for (n, _), reason in sorted(hits.items())]


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
    """Copy the current `target` to `versions_dir/<UTC YYYYMMDDTHHMMSSZ>.md` (`-1`, `-2`, ... if that name is
    taken), then replace `target` with the candidate text via a temporary file. Returns the saved version, or None
    if there was no previous persona. Does not lint."""
    text = candidate.read_text(encoding="utf-8")
    version = None
    if target.exists():
        stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        versions_dir.mkdir(parents=True, exist_ok=True)
        version = versions_dir / f"{stamp}.md"
        for i in range(1, 10_000):
            if not version.exists():
                break
            version = versions_dir / f"{stamp}-{i}.md"
        shutil.copyfile(target, version)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return version
