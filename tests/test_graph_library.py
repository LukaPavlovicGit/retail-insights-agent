from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ria.graph.library import ReportMeta, make_confirm_delete, make_resolve_targets, make_save_report, route_after_resolve
from ria.graph.state import AgentState
from ria.tools import FakeToolBox
from tests.graph_fixtures import make_deps, turn_state

REPORTS = [{"id": 1, "title": "Jeans Q1", "created_at": "2026-04-01", "thread_id": "t1"},
           {"id": 2, "title": "Jeans Q2", "created_at": "2026-07-01", "thread_id": "t1"}]


async def test_save_report_without_analysis(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_save_report(deps)(turn_state("save this"))
    assert out["outcome"] == "ok" and "no analysis" in out["rendered"].lower() and deps.llm.calls == []


async def test_save_report_saves_and_creates_candidate_trio(conn, settings):
    tb = FakeToolBox({"reports_save": [{"ok": True, "id": 7}]})
    deps = make_deps(conn, settings, {"title": [ReportMeta(title="Jeans revenue review", tags=["jeans"])]}, toolbox=tb)
    state = turn_state("save this as a report", last_question="how did jeans do", last_sql="SELECT 1", last_analysis="Jeans grew. Contact jane@x.com")
    out = await make_save_report(deps)(state)
    assert out["outcome"] == "saved" and "#7" in out["rendered"] and "Jeans revenue review" in out["rendered"]
    name, kw = tb.calls[0]
    assert name == "reports_save" and kw["user_id"] == "alice" and kw["thread_id"] == "t1" and kw["body"] == "Jeans grew. Contact [email]"
    pending = deps.trio_store.list(status="pending")
    assert len(pending) == 1 and pending[0].source == "user" and pending[0].question == "how did jeans do"
    assert "[email]" in pending[0].report and "jane@x.com" not in pending[0].report


async def test_resolve_targets_by_scope(conn, settings):
    tb = FakeToolBox({"reports_search": [{"reports": REPORTS}, {"reports": []}, {"reports": REPORTS[:1]}]})
    deps = make_deps(conn, settings, toolbox=tb)
    node = make_resolve_targets(deps)
    conv = await node(turn_state("delete the reports from this conversation", report_filter_scope="conversation"))
    assert len(conv["pending_delete"]) == 2 and tb.calls[0][1] == {"user_id": "alice", "text": None, "thread_id": "t1"}
    assert route_after_resolve(conv) == "confirm_delete"
    none = await node(turn_state("delete reports about Acme", report_filter_scope="text", report_filter_text="Acme"))
    assert none["pending_delete"] is None and "No reports" in none["rendered"] and route_after_resolve(none) == "finalize"
    assert tb.calls[1][1]["text"] == "Acme"
    all_ = await node(turn_state("delete all my reports", report_filter_scope="all"))
    assert len(all_["pending_delete"]) == 1 and tb.calls[2][1] == {"user_id": "alice", "text": None, "thread_id": None}


async def test_resolve_targets_unclear_asks(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_resolve_targets(deps)(turn_state("delete reports", report_filter_scope=None))
    assert out["pending_delete"] is None and "Which reports" in out["rendered"] and deps.toolbox.calls == []


def mini_graph(deps):
    g = StateGraph(AgentState)
    g.add_node("resolve_targets", make_resolve_targets(deps))
    g.add_node("confirm_delete", make_confirm_delete(deps))
    g.add_edge(START, "resolve_targets")
    g.add_conditional_edges("resolve_targets", route_after_resolve, {"confirm_delete": "confirm_delete", "finalize": END, "fail": END})
    g.add_edge("confirm_delete", END)
    return g.compile(checkpointer=InMemorySaver())


async def test_delete_interrupts_then_deletes_on_yes(conn, settings):
    tb = FakeToolBox({"reports_search": [{"reports": REPORTS}], "reports_delete": [{"deleted": 2}]})
    deps = make_deps(conn, settings, toolbox=tb)
    app = mini_graph(deps)
    cfg = {"configurable": {"thread_id": "t1"}}
    first = await app.ainvoke(turn_state("delete them", report_filter_scope="conversation"), cfg)
    payload = first["__interrupt__"][0].value
    assert payload["action"] == "delete_reports" and [r["id"] for r in payload["reports"]] == [1, 2]
    assert ("reports_delete", {"user_id": "alice", "ids": [1, 2]}) not in tb.calls
    final = await app.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert final["outcome"] == "deleted" and "2 report" in final["rendered"] and "Jeans Q1" in final["rendered"]
    assert tb.calls[-1] == ("reports_delete", {"user_id": "alice", "ids": [1, 2]})


async def test_delete_cancelled_on_no(conn, settings):
    tb = FakeToolBox({"reports_search": [{"reports": REPORTS}]})
    deps = make_deps(conn, settings, toolbox=tb)
    app = mini_graph(deps)
    cfg = {"configurable": {"thread_id": "t2"}}
    await app.ainvoke(turn_state("delete them", report_filter_scope="conversation"), cfg)
    final = await app.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert final["outcome"] == "cancelled" and "Nothing was deleted" in final["rendered"]
    assert all(name != "reports_delete" for name, _ in tb.calls)
