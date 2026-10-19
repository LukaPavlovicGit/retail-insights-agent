from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class UserCtx(TypedDict):
    id: str
    name: str
    role: str
    categories: list[str] | None


class AgentState(TypedDict, total=False):
    # conversation-level (survive across turns)
    user: UserCtx
    thread_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    last_question: str | None
    last_sql: str | None
    last_analysis: str | None
    # turn-level (reset by begin_turn)
    turn_id: str
    started_at: str
    t0: float
    question: str
    intent: str
    router_reason: str
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
    return {
        "turn_id": turn_id, "started_at": started_at, "t0": t0, "question": question,
        "intent": "", "router_reason": "", "report_filter_scope": None, "report_filter_text": None,
        "retrieved_trios": [], "sql": None, "sql_attempts": [], "guard_events": [],
        "sql_error": None, "sql_error_kind": None, "repairs": 0,
        "result_rows": None, "result_bytes": 0, "analysis": None, "rendered": None, "pending_delete": None,
        "llm_calls": [], "node_timings": [], "outcome": "", "error": None,
    }
