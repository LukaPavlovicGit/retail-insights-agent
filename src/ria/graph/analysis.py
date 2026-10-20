"""Nodes and edge functions for the data-question branch: retrieve examples, write SQL, guard it, run it, repair it, explain it.

Pipeline: retrieve_trios -> generate_sql -> guard_sql -> execute_sql -> analyze -> render, with repair_sql and give_up
as the failure loop. Each failed attempt is appended to `sql_attempts` with an error `kind` (guard, dry-run
kinds such as syntax or timeout, empty, circuit_open, bytes_cap); `route_after_guard` and `route_after_execute`
read that kind and the repair budget to choose the next node. SQL only reaches BigQuery after `guards.sql_guard`
has rewritten it. Depends on `graph.nodes_common` (LLM calls, prompt context), `graph.prompts`, `golden.retrieve` and `deps.toolbox`.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ria.golden.retrieve import retrieve_trios
from ria.graph.deps import Deps
from ria.graph.nodes_common import base_system, call_llm, scope_text, window
from ria.graph.prompts import (
    ANALYZE_SYSTEM, DATA_SOURCE_DOWN_TEXT, DEPTH_INSTRUCTIONS, FORMAT_INSTRUCTIONS, GAVE_UP_TEXT, NO_DATA_TEXT, NO_DATA_TEXT_UNSCOPED,
    QUERY_TOO_LARGE_TEXT, REPAIR_SYSTEM, SQL_SYSTEM,
)
from ria.guards.sql_guard import guard_sql
from ria.llm.client import strip_fences

# Errors where another LLM rewrite cannot help (data source down, query too big): go straight to give_up.
TERMINAL_KINDS = {"circuit_open", "bytes_cap"}


class SqlPlan(BaseModel):
    """Structured output for `generate_sql` and `repair_sql`: a one-sentence `plan` (not used downstream) and the `sql` text."""

    plan: str = ""
    sql: str


def _attempt(sql: str, error: str | None, kind: str | None, row_count: int = 0, nbytes: int = 0) -> dict:
    """One entry of `sql_attempts`; `error` and `kind` are both None for a successful run."""
    return {"sql": sql, "error": error, "kind": kind, "row_count": row_count, "bytes": nbytes}


def _examples(trios: list[dict], with_sql: bool) -> str:
    """Format retrieved Golden Trios for a prompt: with SQL for writing queries, without it for style guidance in the analysis."""
    if not trios:
        return "(none available)"
    parts = []
    for t in trios:
        block = f"Question: {t['question']}\n"
        if with_sql:
            block += f"SQL:\n{t['sql']}\n"
        block += f"Analyst report:\n{t['report']}"
        parts.append(block)
    return "\n\n".join(parts)


def rows_to_markdown(rows: list[dict], limit: int = 20) -> str:
    """Render result rows as a Markdown table (columns taken from the first row, None as blank), showing at most `limit` rows."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows[:limit]:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    if len(rows) > limit:
        lines.append(f"(showing {limit} of {len(rows)} rows)")
    return "\n".join(lines)


def make_retrieve_trios(deps: Deps):
    """Node `retrieve_trios`: reads `question` and the user's categories, writes `retrieved_trios` (up to 3 promoted
    Golden Trios visible to that user, by embedding similarity). No LLM call, one embedding call. Retrieval is
    best effort: on any error it writes an empty list and a `retrieval:failed` guard event instead of failing the
    turn. Always followed by `generate_sql` (or `fail` if an error is set)."""
    async def retrieve(state: dict) -> dict:
        try:
            trios = await retrieve_trios(deps.trio_store, deps.embedder, state["question"], state["user"]["categories"], k=3)
        except Exception as e:
            return {"retrieved_trios": [], "guard_events": state.get("guard_events", []) + [f"retrieval:failed:{type(e).__name__}"]}
        return {"retrieved_trios": [t.to_dict() for t in trios]}

    return retrieve


def make_generate_sql(deps: Deps):
    """Node `generate_sql`: reads the recent message window, summary, retrieved trios, user scope and schema text;
    writes `sql` (code fences stripped). One LLM call (`SQL_SYSTEM`, structured `SqlPlan`). The scope filter is not
    asked of the model; the guard adds it. Always followed by `guard_sql`."""
    async def generate_sql(state: dict) -> dict:
        ctx = base_system(deps, state)
        system = SQL_SYSTEM.format(
            persona=ctx["persona"], user_name=ctx["user_name"], user_role=ctx["user_role"], scope=ctx["scope"],
            summary_block=ctx["summary_block"], schema_text=deps.schema_text,
            examples=_examples(state.get("retrieved_trios", []), with_sql=True),
        )
        res, calls = await call_llm(deps, state, node="generate_sql", system=system, messages=window(state, deps), schema=SqlPlan)
        return {"sql": strip_fences(res.parsed.sql), "llm_calls": calls}

    return generate_sql


def make_guard_sql(deps: Deps):
    """Node `guard_sql`: reads `sql`, writes the rewritten `sql`, `guard_events`, and on failure `sql_error`,
    `sql_error_kind` and a new `sql_attempts` entry. First runs the static `guard_sql` checks, then a BigQuery
    dry run through the toolbox (syntax and byte-cap check without cost). No LLM call. A static rejection has kind
    "guard"; a dry-run rejection carries the runner's `error_kind`. Next node is chosen by `route_after_guard`."""
    async def guard(state: dict) -> dict:
        sql = state.get("sql") or ""
        events = list(state.get("guard_events", []))
        attempts = list(state.get("sql_attempts", []))
        r = guard_sql(sql, categories=state["user"]["categories"], dataset=deps.settings.dataset,
                      row_cap=deps.settings.row_cap, schema=deps.schema)
        events += r.events
        if not r.ok:
            attempts.append(_attempt(sql, r.reason, "guard"))
            return {"guard_events": events, "sql_attempts": attempts, "sql_error": r.reason, "sql_error_kind": "guard"}
        dry = await deps.toolbox.bq_dry_run(r.sql)
        if not dry.get("ok"):
            kind = dry.get("error_kind") or "dry_run"
            events.append(f"dryrun:rejected:{kind}")
            attempts.append(_attempt(r.sql, dry.get("error"), kind, 0, int(dry.get("bytes") or 0)))
            return {"sql": r.sql, "guard_events": events, "sql_attempts": attempts, "sql_error": dry.get("error"), "sql_error_kind": kind}
        events.append("dryrun:ok")
        return {"sql": r.sql, "guard_events": events, "sql_error": None, "sql_error_kind": None}

    return guard


def make_execute_sql(deps: Deps):
    """Node `execute_sql`: runs the guarded `sql` with `bq_query` and writes `result_rows`, `result_bytes` and a
    `sql_attempts` entry. A query that succeeds but returns zero rows is recorded as kind "empty" so it can be
    repaired once (filters may be too strict) rather than analysed. No LLM call. Next node is chosen by `route_after_execute`."""
    async def execute_sql(state: dict) -> dict:
        sql = state["sql"]
        attempts = list(state.get("sql_attempts", []))
        res = await deps.toolbox.bq_query(sql)
        nbytes = int(res.get("bytes") or 0)
        if not res.get("ok"):
            kind = res.get("error_kind") or "unknown"
            attempts.append(_attempt(sql, res.get("error"), kind, 0, nbytes))
            return {"sql_attempts": attempts, "sql_error": res.get("error"), "sql_error_kind": kind, "result_bytes": nbytes}
        rows = res.get("rows") or []
        if not rows:
            attempts.append(_attempt(sql, "Query returned no rows", "empty", 0, nbytes))
            return {"sql_attempts": attempts, "sql_error": "Query returned no rows", "sql_error_kind": "empty", "result_bytes": nbytes, "result_rows": []}
        attempts.append(_attempt(sql, None, None, len(rows), nbytes))
        return {"sql_attempts": attempts, "sql_error": None, "sql_error_kind": None, "result_rows": rows, "result_bytes": nbytes}

    return execute_sql


def _can_repair(state: dict, max_repairs: int) -> bool:
    """True while fewer than `max_repairs` repair rewrites have been used this turn."""
    return int(state.get("repairs", 0)) < max_repairs


def route_after_guard(state: dict, max_repairs: int = 2) -> str:
    """Edge after `guard_sql`: error set -> fail; no SQL error -> execute_sql; terminal kind (circuit_open,
    bytes_cap) -> give_up; any other error -> repair_sql while the repair budget lasts, else give_up."""
    if state.get("error"):
        return "fail"
    kind = state.get("sql_error_kind")
    if kind is None:
        return "execute_sql"
    if kind in TERMINAL_KINDS:
        return "give_up"
    return "repair_sql" if _can_repair(state, max_repairs) else "give_up"


def route_after_execute(state: dict, max_repairs: int = 2) -> str:
    """Edge after `execute_sql`: error set -> fail; success -> analyze; terminal kind -> give_up; a second "empty"
    result -> give_up (an empty answer gets one retry only); any other error -> repair_sql while budget lasts, else give_up."""
    if state.get("error"):
        return "fail"
    kind = state.get("sql_error_kind")
    if kind is None:
        return "analyze"
    if kind in TERMINAL_KINDS:
        return "give_up"
    if kind == "empty":
        empties = sum(1 for a in state.get("sql_attempts", []) if a.get("kind") == "empty")
        if empties > 1:
            return "give_up"
    return "repair_sql" if _can_repair(state, max_repairs) else "give_up"


def make_repair_sql(deps: Deps):
    """Node `repair_sql`: reads every entry in `sql_attempts` (SQL plus its error or row count) and the original
    question; writes a new `sql`, increments `repairs`, and clears the error fields. One LLM call (`REPAIR_SYSTEM`,
    structured `SqlPlan`). The new SQL goes back through `guard_sql`, never straight to BigQuery."""
    async def repair_sql(state: dict) -> dict:
        ctx = base_system(deps, state)
        blocks = []
        for i, a in enumerate(state.get("sql_attempts", []), 1):
            result = a["error"] or f"{a['row_count']} rows"
            blocks.append(f"Attempt {i}:\n{a['sql']}\nResult: {result}")
        history = "\n\n".join(blocks)
        system = REPAIR_SYSTEM.format(persona=ctx["persona"], user_name=ctx["user_name"], user_role=ctx["user_role"],
                                      scope=ctx["scope"], schema_text=deps.schema_text, history=history)
        res, calls = await call_llm(deps, state, node="repair_sql", system=system,
                                    messages=[HumanMessage(content=f"Original question: {state['question']}")], schema=SqlPlan)
        return {"sql": strip_fences(res.parsed.sql), "repairs": int(state.get("repairs", 0)) + 1,
                "sql_error": None, "sql_error_kind": None, "llm_calls": calls}

    return repair_sql


def make_give_up(deps: Deps):
    """Node `give_up`: reads `sql_error_kind` and `sql_attempts`, writes a canned `rendered` message and `outcome`.
    Empty result -> "no_data" (scope-aware text); circuit_open -> "error"; bytes_cap -> "gave_up" with the cap
    detail; anything else -> "gave_up" listing each attempt's error. No LLM call, so it works when the model or
    warehouse is struggling. Always followed by `finalize`."""
    async def give_up(state: dict) -> dict:
        kind = state.get("sql_error_kind")
        attempts = state.get("sql_attempts", [])
        if kind == "empty":
            sql = state.get("sql") or ""
            if state["user"]["categories"] is None:
                text = NO_DATA_TEXT_UNSCOPED.format(sql=sql)
            else:
                text = NO_DATA_TEXT.format(scope=scope_text(state), sql=sql)
            return {"rendered": text, "outcome": "no_data"}
        if kind == "circuit_open":
            return {"rendered": DATA_SOURCE_DOWN_TEXT, "outcome": "error"}
        if kind == "bytes_cap":
            return {"rendered": QUERY_TOO_LARGE_TEXT.format(detail=state.get("sql_error") or "byte cap"), "outcome": "gave_up"}
        listing = "\n".join(f"{i}. {a.get('error') or 'no rows'}" for i, a in enumerate(attempts, 1)) or "(no attempts recorded)"
        return {"rendered": GAVE_UP_TEXT.format(n=len(attempts), attempts=listing), "outcome": "gave_up"}

    return give_up


def make_analyze(deps: Deps):
    """Node `analyze`: reads `result_rows` (first 50, as JSON), `sql`, the question, the user's preferences and
    retrieved trios (for style only, without their SQL); writes `analysis`, the model's written interpretation.
    One LLM call (`ANALYZE_SYSTEM`), with depth and format taken from the user's stored preferences. Always followed by `render`."""
    async def analyze(state: dict) -> dict:
        ctx = base_system(deps, state)
        prefs = ctx["prefs_obj"]
        system = ANALYZE_SYSTEM.format(
            persona=ctx["persona"], user_name=ctx["user_name"], user_role=ctx["user_role"], scope=ctx["scope"],
            prefs=ctx["prefs"], depth_instruction=DEPTH_INSTRUCTIONS[prefs.depth],
            format_instruction=FORMAT_INSTRUCTIONS[prefs.format],
            examples=_examples(state.get("retrieved_trios", []), with_sql=False),
        )
        rows = state.get("result_rows") or []
        data = json.dumps(rows[:50], default=str)
        msgs = window(state, deps)[:-1] + [HumanMessage(content=f"Question: {state['question']}\n\nSQL used:\n{state.get('sql')}\n\nResult rows (JSON, up to 50):\n{data}")]
        res, calls = await call_llm(deps, state, node="analyze", system=system, messages=msgs)
        return {"analysis": res.text.strip(), "llm_calls": calls}

    return analyze


def make_render(deps: Deps):
    """Node `render`: builds the final `rendered` text from `analysis`, appending a Markdown table when the user's
    format is "table" and the SQL when their depth is "deep". Preferences are read fresh from the store, so a
    change made earlier in the same conversation applies. No LLM call. Always followed by `finalize`."""
    async def render(state: dict) -> dict:
        prefs = deps.prefs_store.get(state["user"]["id"])
        parts = [state.get("analysis") or ""]
        rows = state.get("result_rows") or []
        if prefs.format == "table" and rows:
            parts.append(rows_to_markdown(rows))
        if prefs.depth == "deep" and state.get("sql"):
            parts.append(f"SQL used:\n```sql\n{state['sql']}\n```")
        return {"rendered": "\n\n".join(p for p in parts if p), "outcome": "ok"}

    return render
