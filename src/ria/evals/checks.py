"""Eval case format and the deterministic checks applied to a turn's trace.

The checks need no LLM: they compare the recorded intent, outcome, repair count and final SQL with what the case expects,
and confirm the user's scope appears in the SQL that ran. `evals.runner` loads cases from YAML and calls `check_case`. Depends on `observability.trace`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ria.observability.trace import TurnTrace


@dataclass
class EvalCase:
    """One scripted question for one demo `user`, with the expected `intent` and `outcome`, `tables` the final SQL
    must reference, a `max_repairs` allowance, and `must_not_match` regexes that must not appear in the answer
    (for example email addresses)."""

    id: str
    user: str
    question: str
    intent: str
    tables: list[str] = field(default_factory=list)
    outcome: str = "ok"
    max_repairs: int = 1
    must_not_match: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    """Parse a YAML list of cases; `id`, `user`, `question` and `intent` are required, the rest default (outcome "ok", one repair)."""
    raw = yaml.safe_load(Path(path).read_text()) or []
    return [EvalCase(
        id=c["id"], user=c["user"], question=c["question"], intent=c["intent"], tables=list(c.get("tables") or []),
        outcome=c.get("outcome", "ok"), max_repairs=int(c.get("max_repairs", 1)),
        must_not_match=list(c.get("must_not_match") or []),
    ) for c in raw]


def check_case(case: EvalCase, trace: TurnTrace, rendered: str, user_scope: list[str] | None, scope_column: str = "brand") -> list[str]:
    """Return a list of failure messages (empty means the case passed).

    Checks intent, outcome, repairs within `max_repairs`, expected tables in the last SQL attempt, forbidden patterns
    in `rendered`, and, for scoped users on analysis cases, a `<scope_column> IN (...)` predicate (also the semi-join form) and every
    scope value as a literal, with `\'` accepted for apostrophes. The scope check is a text match on the final SQL, a backstop for the guard rather than a proof of it.
    """
    failures: list[str] = []
    if trace.intent != case.intent:
        failures.append(f"intent: expected {case.intent}, got {trace.intent}")
    if trace.outcome != case.outcome:
        failures.append(f"outcome: expected {case.outcome}, got {trace.outcome}")
    if trace.repairs > case.max_repairs:
        failures.append(f"repairs: {trace.repairs} > {case.max_repairs}")
    final_sql = trace.sql_attempts[-1]["sql"] if trace.sql_attempts else ""
    for table in case.tables:
        if not re.search(rf"\b{re.escape(table)}\b", final_sql):
            failures.append(f"tables: {table} not in final SQL")
    for pattern in case.must_not_match:
        if re.search(pattern, rendered or ""):
            failures.append(f"must_not_match: /{pattern}/ matched the answer")
    if case.intent == "analysis" and user_scope and final_sql:
        if not re.search(rf"{re.escape(scope_column)}`?\s+IN\s*\(", final_sql, re.I):
            failures.append(f"scope: final SQL has no {scope_column} IN predicate")
        for value in user_scope:
            if f"'{value.replace(chr(39), chr(92) + chr(39))}'" not in final_sql and f"'{value}'" not in final_sql:
                failures.append(f"scope: {scope_column} '{value}' missing from final SQL")
    return failures
