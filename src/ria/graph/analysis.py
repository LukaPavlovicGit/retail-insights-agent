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

TERMINAL_KINDS = {"circuit_open", "bytes_cap"}


class SqlPlan(BaseModel):
    plan: str = ""
    sql: str


def _attempt(sql: str, error: str | None, kind: str | None, row_count: int = 0, nbytes: int = 0) -> dict:
    return {"sql": sql, "error": error, "kind": kind, "row_count": row_count, "bytes": nbytes}


def _examples(trios: list[dict], with_sql: bool) -> str:
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
    async def retrieve(state: dict) -> dict:
        try:
            trios = await retrieve_trios(deps.trio_store, deps.embedder, state["question"], state["user"]["categories"], k=3)
        except Exception as e:
            return {"retrieved_trios": [], "guard_events": state.get("guard_events", []) + [f"retrieval:failed:{type(e).__name__}"]}
        return {"retrieved_trios": [t.to_dict() for t in trios]}

    return retrieve


def make_generate_sql(deps: Deps):
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
    return int(state.get("repairs", 0)) < max_repairs


def route_after_guard(state: dict, max_repairs: int = 2) -> str:
    if state.get("error"):
        return "fail"
    kind = state.get("sql_error_kind")
    if kind is None:
        return "execute_sql"
    if kind in TERMINAL_KINDS:
        return "give_up"
    return "repair_sql" if _can_repair(state, max_repairs) else "give_up"


def route_after_execute(state: dict, max_repairs: int = 2) -> str:
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
