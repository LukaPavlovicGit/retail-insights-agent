"""Assemble the per-turn LangGraph: every node, every edge, and the routing tables.

Flow of one turn: begin_turn -> router -> one intent branch -> finalize -> update_memory -> write_trace -> END.
The analysis branch is retrieve_trios -> generate_sql -> guard_sql -> execute_sql -> analyze -> render, with
guard_sql and execute_sql able to loop through repair_sql or stop at give_up. Any node can divert to `fail`
by setting `state["error"]`; all terminal branches pass through finalize so masking, memory and tracing always run.
This module only wires things; node behaviour is in `analysis`, `library`, `router`, `schema_qa` and `lifecycle`.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ria.graph.analysis import (
    make_analyze, make_execute_sql, make_generate_sql, make_give_up, make_guard_sql, make_render, make_repair_sql,
    make_retrieve_trios, route_after_execute, route_after_guard,
)
from ria.graph.deps import Deps
from ria.graph.library import make_confirm_delete, make_resolve_targets, make_save_report, route_after_resolve
from ria.graph.lifecycle import make_begin_turn, make_fail, make_finalize, make_update_memory, make_write_trace
from ria.graph.nodes_common import after, route_intent, timed
from ria.graph.router import make_ack_preference, make_refuse, make_router
from ria.graph.schema_qa import make_schema_answer
from ria.graph.state import AgentState

# Router intent -> first node of that branch ("fail" is returned by route_intent when state["error"] is set).
INTENT_ROUTES = {
    "analysis": "retrieve_trios",
    "schema_question": "schema_answer",
    "report_save": "save_report",
    "report_delete": "resolve_targets",
    "preference_update": "ack_preference",
    "off_topic": "refuse",
    "fail": "fail",
}


def build_graph(deps: Deps, checkpointer=None):
    """Compile the turn graph with `deps` closed over by every node.

    Each node is wrapped by `timed` (records node timings and turns LLM-budget and model-outage errors into
    `state["error"]`). `max_repairs` from settings is bound into the two repair edge functions. A checkpointer is
    required for the delete confirmation, because `interrupt` pauses the turn and resumes from saved state;
    chat uses SQLite, evals use an in-memory saver, and `None` is fine for tests that never interrupt.
    """
    s = deps.settings
    g = StateGraph(AgentState)
    factories = {
        "begin_turn": make_begin_turn, "router": make_router, "refuse": make_refuse, "ack_preference": make_ack_preference,
        "schema_answer": make_schema_answer, "retrieve_trios": make_retrieve_trios, "generate_sql": make_generate_sql,
        "guard_sql": make_guard_sql, "execute_sql": make_execute_sql, "repair_sql": make_repair_sql, "give_up": make_give_up,
        "analyze": make_analyze, "render": make_render, "save_report": make_save_report,
        "resolve_targets": make_resolve_targets, "confirm_delete": make_confirm_delete, "fail": make_fail,
        "finalize": make_finalize, "update_memory": make_update_memory, "write_trace": make_write_trace,
    }
    for name, factory in factories.items():
        g.add_node(name, timed(name, factory(deps), deps.clock))

    def after_guard(state: dict) -> str:
        return route_after_guard(state, max_repairs=s.max_repairs)

    def after_execute(state: dict) -> str:
        return route_after_execute(state, max_repairs=s.max_repairs)

    g.add_edge(START, "begin_turn")
    g.add_conditional_edges("begin_turn", after("router"), {"router": "router", "fail": "fail"})
    g.add_conditional_edges("router", route_intent, INTENT_ROUTES)
    g.add_conditional_edges("retrieve_trios", after("generate_sql"), {"generate_sql": "generate_sql", "fail": "fail"})
    g.add_conditional_edges("generate_sql", after("guard_sql"), {"guard_sql": "guard_sql", "fail": "fail"})
    g.add_conditional_edges("guard_sql", after_guard, {"execute_sql": "execute_sql", "repair_sql": "repair_sql", "give_up": "give_up", "fail": "fail"})
    g.add_conditional_edges("execute_sql", after_execute, {"analyze": "analyze", "repair_sql": "repair_sql", "give_up": "give_up", "fail": "fail"})
    g.add_conditional_edges("repair_sql", after("guard_sql"), {"guard_sql": "guard_sql", "fail": "fail"})
    g.add_conditional_edges("analyze", after("render"), {"render": "render", "fail": "fail"})
    g.add_conditional_edges("schema_answer", after("finalize"), {"finalize": "finalize", "fail": "fail"})
    g.add_conditional_edges("save_report", after("finalize"), {"finalize": "finalize", "fail": "fail"})
    g.add_conditional_edges("resolve_targets", route_after_resolve, {"confirm_delete": "confirm_delete", "finalize": "finalize", "fail": "fail"})
    for name in ("render", "refuse", "ack_preference", "give_up", "confirm_delete", "fail"):
        g.add_edge(name, "finalize")
    g.add_edge("finalize", "update_memory")
    g.add_edge("update_memory", "write_trace")
    g.add_edge("write_trace", END)
    return g.compile(checkpointer=checkpointer)
