from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ria.config import Settings
from ria.graph.analysis import SqlPlan
from ria.graph.build import build_graph
from ria.graph.library import ReportMeta
from ria.graph.router import RouteDecision
from ria.graph.run import run_turn
from ria.llm.client import LLMUnavailable
from ria.tools import FakeToolBox
from tests.graph_fixtures import ALICE, make_deps

DS = "bigquery-public-data.thelook_ecommerce"
GOOD_SQL = (f"SELECT p.category, SUM(oi.sale_price) AS revenue FROM `{DS}.order_items` oi "
            f"JOIN `{DS}.products` p ON p.id = oi.product_id GROUP BY 1 ORDER BY 2 DESC")
ROWS = [{"category": "Jeans", "revenue": 1200.5}, {"category": "Pants", "revenue": 800.0}]
OK_QUERY = {"ok": True, "rows": ROWS, "row_count": 2, "bytes": 99, "error": None, "error_kind": None}
BAD_QUERY = {"ok": False, "rows": [], "row_count": 0, "bytes": 0, "error": "Syntax error", "error_kind": "syntax"}
ANALYSIS = RouteDecision(intent="analysis", reason="data question")


def app_for(conn, settings, script, toolbox=None, checkpointer=None):
    deps = make_deps(conn, settings, script, toolbox=toolbox)
    return build_graph(deps, checkpointer or InMemorySaver()), deps


async def test_analysis_happy_path(conn, settings):
    tb = FakeToolBox({"bq_query": [OK_QUERY]})
    app, deps = app_for(conn, settings, {"router": [ANALYSIS], "generate_sql": [SqlPlan(sql=GOOD_SQL)], "analyze": ["Jeans lead with 1,200."]}, tb)
    r = await run_turn(app, ALICE, "t1", "which category leads revenue?", new_turn_id=lambda: "turn-1")
    assert r.outcome == "ok" and r.rendered.startswith("Jeans lead") and r.interrupt is None
    t = deps.trace_store.get("turn-1")
    assert t.intent == "analysis" and t.outcome == "ok" and len(t.llm_calls) == 3 and t.repairs == 0
    assert "scope:injected" in t.guard_events and t.bq_bytes == 99
    assert [n["node"] for n in t.node_timings][:3] == ["begin_turn", "router", "retrieve_trios"]
    assert "IN ('Levi\\'s', 'Carhartt', 'Dockers', 'Wrangler')" in tb.calls[-1][1]["sql"]


async def test_repair_loop_recovers(conn, settings):
    tb = FakeToolBox({"bq_query": [BAD_QUERY, OK_QUERY]})
    app, deps = app_for(conn, settings, {"router": [ANALYSIS], "generate_sql": [SqlPlan(sql=GOOD_SQL)], "repair_sql": [SqlPlan(sql=GOOD_SQL)], "analyze": ["Fixed."]}, tb)
    r = await run_turn(app, ALICE, "t1", "q", new_turn_id=lambda: "turn-2")
    t = deps.trace_store.get("turn-2")
    assert r.outcome == "ok" and t.repairs == 1 and len(t.llm_calls) == 4


async def test_gives_up_after_repair_budget(conn, settings):
    tb = FakeToolBox({"bq_query": [BAD_QUERY, BAD_QUERY, BAD_QUERY]})
    app, deps = app_for(conn, settings, {"router": [ANALYSIS], "generate_sql": [SqlPlan(sql=GOOD_SQL)], "repair_sql": [SqlPlan(sql=GOOD_SQL), SqlPlan(sql=GOOD_SQL)]}, tb)
    r = await run_turn(app, ALICE, "t1", "q", new_turn_id=lambda: "turn-3")
    t = deps.trace_store.get("turn-3")
    assert r.outcome == "gave_up" and "3 attempts" in r.rendered and t.repairs == 2 and len(t.llm_calls) == 4


async def test_off_topic_is_refused_with_one_call(conn, settings):
    app, deps = app_for(conn, settings, {"router": [RouteDecision(intent="off_topic", reason="poem")]})
    r = await run_turn(app, ALICE, "t1", "write me a poem", new_turn_id=lambda: "turn-4")
    assert r.outcome == "refused" and len(deps.trace_store.get("turn-4").llm_calls) == 1


async def test_llm_outage_is_graceful(conn, settings):
    app, deps = app_for(conn, settings, {"router": [LLMUnavailable("quota")]})
    r = await run_turn(app, ALICE, "t1", "q", new_turn_id=lambda: "turn-5")
    assert r.outcome == "error" and "unavailable" in r.rendered.lower()
    assert deps.trace_store.get("turn-5").error.startswith("LLMUnavailable")


async def test_preference_then_table_render(conn, settings):
    tb = FakeToolBox({"bq_query": [OK_QUERY]})
    app, deps = app_for(conn, settings, {
        "router": [RouteDecision(intent="preference_update", reason="format"), ANALYSIS],
        "generate_sql": [SqlPlan(sql=GOOD_SQL)], "analyze": ["Jeans lead."]}, tb)
    r1 = await run_turn(app, ALICE, "t1", "use tables from now on")
    assert "Format: table" in r1.rendered
    r2 = await run_turn(app, ALICE, "t1", "which category leads?")
    assert "| category | revenue |" in r2.rendered


async def test_save_then_delete_with_confirmation_survives_restart(conn, settings, tmp_path):
    tb = FakeToolBox({"bq_query": [OK_QUERY], "reports_save": [{"ok": True, "id": 3}],
                      "reports_search": [{"reports": [{"id": 3, "title": "Jeans review", "created_at": "2026-10-16", "thread_id": "t9"}]}],
                      "reports_delete": [{"deleted": 1}]})
    script = {"router": [ANALYSIS, RouteDecision(intent="report_save", reason="save"),
                         RouteDecision(intent="report_delete", reason="del", report_filter_scope="conversation")],
              "generate_sql": [SqlPlan(sql=GOOD_SQL)], "analyze": ["Jeans lead."], "title": [ReportMeta(title="Jeans review", tags=["jeans"])]}
    deps = make_deps(conn, settings, script, toolbox=tb)
    db_file = tmp_path / "ckpt.db"
    async with AsyncSqliteSaver.from_conn_string(str(db_file)) as cp:
        app = build_graph(deps, cp)
        await run_turn(app, ALICE, "t9", "which category leads?")
        saved = await run_turn(app, ALICE, "t9", "save this as a report")
        assert saved.outcome == "saved" and "#3" in saved.rendered
        pending = await run_turn(app, ALICE, "t9", "delete the reports from this conversation")
        assert pending.outcome == "needs_confirmation" and pending.interrupt["reports"][0]["id"] == 3
    # process restart: new checkpointer on the same file, resume the pending confirmation
    async with AsyncSqliteSaver.from_conn_string(str(db_file)) as cp2:
        app2 = build_graph(deps, cp2)
        done = await run_turn(app2, ALICE, "t9", "", resume={"confirmed": True})
    assert done.outcome == "deleted" and "Jeans review" in done.rendered
    assert tb.calls[-1] == ("reports_delete", {"user_id": "alice", "ids": [3]})
    assert deps.trace_store.get(done.turn_id).outcome == "deleted"


async def test_context_summary_kicks_in(conn, tmp_path):
    settings = Settings(db_path=tmp_path / "x.db", data_dir=tmp_path, embedding_dim=32, window_turns=1)
    routes = [RouteDecision(intent="off_topic", reason="x")] * 4
    app, deps = app_for(conn, settings, {"router": routes, "summarize": ["They chatted."]})
    for i in range(3):
        await run_turn(app, ALICE, "t1", f"hi {i}", new_turn_id=lambda i=i: f"turn-{i}")
    state = await app.aget_state({"configurable": {"thread_id": "t1"}})
    assert state.values["summary"] == "They chatted."
    assert len(state.values["messages"]) <= 2 * settings.window_turns + 2
    assert any(c["node"] == "summarize" for c in deps.trace_store.get("turn-2").llm_calls)
