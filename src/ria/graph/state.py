"""State schema for the turn graph.

`AgentState` is what LangGraph checkpoints per thread. Conversation-level keys persist across turns; turn-level keys are
reset by `begin_turn` through `new_turn_fields`. Nodes return partial dicts of these keys, and edge functions
read them (mostly `error`, `intent`, `sql_error_kind`, `repairs`, `pending_delete`). Has no dependencies on the rest of the package.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class UserCtx(TypedDict):
    """The signed-in demo user, as returned by `db.get_user`. `scope` is the allowed values of
    `Settings.scope_column` (brands by default), or None for an unrestricted user. The SQL guard enforces it; prompts only describe it."""

    id: str
    name: str
    role: str
    scope: list[str] | None


class AgentState(TypedDict, total=False):
    """Full graph state (all keys optional; read with `.get`). `messages` uses LangGraph's `add_messages` reducer, so
    returned messages are appended, and `RemoveMessage` entries delete older ones. `last_*` hold the most recent
    successful analysis for "save this as a report". `sql_attempts` entries are `{sql, error, kind, row_count, bytes}`;
    `llm_calls` entries are `LLMResult.ledger()` dicts; `error` being set sends the turn to `fail`."""

    # conversation-level (survive across turns)
    user: UserCtx
    thread_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    last_question: str | None
    last_sql: str | None
    last_analysis: str | None
    last_turn_id: str | None
    # turn-level (reset by begin_turn)
    turn_id: str
    started_at: str
    t0: float
    question: str
    intent: str
    router_reason: str
    command: dict | None
    correction_of: str | None
    report_filter_scope: str | None
    report_filter_text: str | None
    retrieved_trios: list[dict]
    sql: str | None
    sql_attempts: list[dict]
    guard_events: list[str]
    sql_error: str | None
    sql_error_kind: str | None
    repairs: int
    result_rows: list[dict] | None
    result_bytes: int
    analysis: str | None
    rendered: str | None
    pending_delete: list[dict] | None
    llm_calls: list[dict]
    node_timings: list[dict]
    outcome: str
    error: str | None


def new_turn_fields(turn_id: str, question: str, started_at: str, t0: float) -> dict:
    """Fresh values for every turn-level key, so nothing from the previous turn (SQL, errors, repairs, rendered text) leaks into this one."""
    return {
        "turn_id": turn_id, "started_at": started_at, "t0": t0, "question": question,
        "intent": "", "router_reason": "", "command": None, "correction_of": None, "report_filter_scope": None, "report_filter_text": None,
        "retrieved_trios": [], "sql": None, "sql_attempts": [], "guard_events": [],
        "sql_error": None, "sql_error_kind": None, "repairs": 0,
        "result_rows": None, "result_bytes": 0, "analysis": None, "rendered": None, "pending_delete": None,
        "llm_calls": [], "node_timings": [], "outcome": "", "error": None,
    }
