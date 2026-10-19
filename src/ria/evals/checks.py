from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ria.observability.trace import TurnTrace


@dataclass
class EvalCase:
    id: str
    user: str
    question: str
    intent: str
    tables: list[str] = field(default_factory=list)
    outcome: str = "ok"
    max_repairs: int = 1
    must_not_match: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    return [EvalCase(
        id=c["id"], user=c["user"], question=c["question"], intent=c["intent"], tables=list(c.get("tables") or []),
        outcome=c.get("outcome", "ok"), max_repairs=int(c.get("max_repairs", 1)),
        must_not_match=list(c.get("must_not_match") or []),
    ) for c in raw]


def check_case(case: EvalCase, trace: TurnTrace, rendered: str, user_categories: list[str] | None) -> list[str]:
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
    if case.intent == "analysis" and user_categories and final_sql:
        if not re.search(r"category`?\s+IN\s*\(", final_sql, re.I):
            failures.append("scope: final SQL has no category IN predicate")
        for value in user_categories:
            if f"'{value.replace(chr(39), chr(92) + chr(39))}'" not in final_sql and f"'{value}'" not in final_sql:
                failures.append(f"scope: category '{value}' missing from final SQL")
    return failures
