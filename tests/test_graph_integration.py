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
    assert [n["node"] for n in t.node_timings][:4] == ["begin_turn", "pre_router", "router", "retrieve_trios"]
    assert "IN ('Levi\\'s', 'Carhartt', 'Dockers', 'Wrangler')" in tb.calls[-1][1]["sql"]


async def test_invalid_org_defaults_goes_through_error_path(conn, settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.org_defaults_path.write_text("min_group_size: 0\n")
    app, deps = app_for(conn, settings, {"router": [ANALYSIS], "generate_sql": [SqlPlan(sql=GOOD_SQL)]})
    r = await run_turn(app, ALICE, "t1", "which category leads revenue?", new_turn_id=lambda: "turn-od")
    assert r.outcome == "error" and "org defaults file is invalid" in r.rendered and "min_group_size" in r.rendered
    assert deps.trace_store.get("turn-od") is not None


async def test_empty_result_ends_as_no_data_without_a_repair(conn, settings):
    """Live run 2026-10-26: 'How did Calvin Klein sell last month?' (outside Alice's scope) returned no rows and the
    one repair dropped the brand and the month, answering a different question. An empty result is now final."""
    empty = {"ok": True, "rows": [], "row_count": 0, "bytes": 5, "error": None, "error_kind": None}
    tb = FakeToolBox({"bq_query": [empty, OK_QUERY]})
    app, deps = app_for(conn, settings, {"router": [ANALYSIS], "generate_sql": [SqlPlan(sql=GOOD_SQL), SqlPlan(sql=GOOD_SQL)],
                                         "analyze": ["should not run"]}, tb)
    r = await run_turn(app, ALICE, "t1", "how did Calvin Klein sell last month?", new_turn_id=lambda: "e1")
    assert r.outcome == "no_data" and "returned no rows" in r.rendered and "Levi's, Carhartt, Dockers, Wrangler" in r.rendered
    t = deps.trace_store.get("e1")
    assert t.repairs == 0 and len(t.sql_attempts) == 1 and [c["node"] for c in t.llm_calls] == ["router", "generate_sql"]
    assert len([c for c in tb.calls if c[0] == "bq_query"]) == 1


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
    # the second turn's trace lists only its own nodes, not the first turn's as well
    nodes = [n["node"] for n in deps.trace_store.get(r2.turn_id).node_timings]
    assert nodes.count("begin_turn") == 1 and nodes[0] == "begin_turn" and "generate_sql" in nodes


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


def models_down(conn, settings, toolbox=None, settings_override=None):
    app, deps = app_for(conn, settings_override or settings, {}, toolbox)
    deps.llm.default = LLMUnavailable("down")
    return app, deps


async def test_commands_work_with_both_models_down(conn, settings):
    report = {"id": 1, "title": "Jeans review", "question": "q", "sql": "SELECT 1", "body": "Jeans grew.", "created_at": "2026-10-16"}
    tb = FakeToolBox({"reports_search": [{"reports": [{"id": 1, "title": "Jeans review", "created_at": "2026-10-16", "thread_id": "t1"}]}],
                      "reports_get": [{"report": report}]})
    app, deps = models_down(conn, settings, tb)
    expected = [("/good", "ok", "Nothing to rate yet"), ("/reports", "listed", "Jeans review"), ("/report 1", "shown", "Jeans grew."),
                ("/schema", "ok", "order_items"), ("/help", "help", "/reports")]
    for i, (text, outcome, needle) in enumerate(expected):
        r = await run_turn(app, ALICE, "t1", text, new_turn_id=lambda i=i: f"cmd-{i}")
        assert r.outcome == outcome and needle in r.rendered, text
        t = deps.trace_store.get(f"cmd-{i}")
        assert t is not None and t.outcome == outcome and t.llm_calls == [] and t.error is None
        assert "router" not in [n["node"] for n in t.node_timings]
    assert deps.llm.calls == []


async def test_delete_command_asks_for_confirmation_with_models_down(conn, settings):
    tb = FakeToolBox({"reports_search": [{"reports": [{"id": 3, "title": "Jeans review", "created_at": "2026-10-16", "thread_id": "t1"}]}],
                      "reports_delete": [{"deleted": 1}]})
    app, deps = models_down(conn, settings, tb)
    pending = await run_turn(app, ALICE, "t1", "/delete from this conversation")
    assert pending.outcome == "needs_confirmation" and pending.interrupt["reports"][0]["id"] == 3
    assert tb.calls[0] == ("reports_search", {"user_id": "alice", "text": None, "thread_id": "t1"})
    done = await run_turn(app, ALICE, "t1", "", resume={"confirmed": True})
    assert done.outcome == "deleted" and deps.llm.calls == []


async def test_command_routes_with_overflowing_window_and_models_down(conn, tmp_path):
    settings = Settings(db_path=tmp_path / "x.db", data_dir=tmp_path, embedding_dim=32, window_turns=1)
    app, deps = models_down(conn, settings)
    for i in range(3):
        await run_turn(app, ALICE, "t1", "/help", new_turn_id=lambda i=i: f"h-{i}")
    r = await run_turn(app, ALICE, "t1", "/reports", new_turn_id=lambda: "after")
    assert r.outcome == "listed"
    t = deps.trace_store.get("after")
    assert "summary:skipped_llm_unavailable" in t.guard_events and t.error is None


async def test_rating_applies_to_last_answer_not_to_listing(conn, settings):
    tb = FakeToolBox({"bq_query": [OK_QUERY]})
    app, deps = app_for(conn, settings, {"router": [ANALYSIS], "generate_sql": [SqlPlan(sql=GOOD_SQL)], "analyze": ["Jeans lead."]}, tb)
    await run_turn(app, ALICE, "t1", "which category leads?", new_turn_id=lambda: "a1")
    listing = await run_turn(app, ALICE, "t1", "/reports", new_turn_id=lambda: "l1")
    assert listing.outcome == "listed"
    rated = await run_turn(app, ALICE, "t1", "/good nice", new_turn_id=lambda: "r1")
    assert rated.outcome == "rated"
    assert (deps.trace_store.get("a1").rating, deps.trace_store.get("a1").rating_note) == (1, "nice")
    assert deps.trace_store.get("l1").rating is None and deps.trace_store.get("r1").rating is None
    # rating again still targets the analysis turn
    await run_turn(app, ALICE, "t1", "/bad", new_turn_id=lambda: "r2")
    assert deps.trace_store.get("a1").rating == -1


async def test_good_as_first_turn_does_not_become_the_rating_target(conn, settings):
    app, deps = models_down(conn, settings)
    first = await run_turn(app, ALICE, "t1", "/good", new_turn_id=lambda: "g1")
    second = await run_turn(app, ALICE, "t1", "/good", new_turn_id=lambda: "g2")
    assert "Nothing to rate yet" in first.rendered and "Nothing to rate yet" in second.rendered
    assert deps.trace_store.get("g1").rating is None


async def test_correction_marks_previous_turn(conn, settings):
    tb = FakeToolBox({"bq_query": [OK_QUERY, OK_QUERY]})
    app, deps = app_for(conn, settings, {
        "router": [ANALYSIS, RouteDecision(intent="analysis", reason="redo", is_correction=True)],
        "generate_sql": [SqlPlan(sql=GOOD_SQL), SqlPlan(sql=GOOD_SQL)], "analyze": ["First.", "Second."]}, tb)
    await run_turn(app, ALICE, "t1", "which category leads?", new_turn_id=lambda: "c1")
    await run_turn(app, ALICE, "t1", "no, I meant by brand", new_turn_id=lambda: "c2")
    assert deps.trace_store.get("c1").corrected and not deps.trace_store.get("c2").corrected


async def test_app_db_and_checkpointer_on_their_own_files_never_lock(settings):
    """Regression for a live-run failure: with the checkpointer on the app's sqlite file, `write_trace` (a blocking
    write on the event loop thread) waited for a write lock held by the checkpointer's aiosqlite thread, whose commit
    needed the blocked loop. After the 5 s busy timeout every such turn ended `OperationalError: database is locked`
    (25 of 40 command turns in the reproduction). Separate files remove the shared lock."""
    from ria import db
    conn = db.connect(settings.db_path)
    db.init_schema(conn, settings.embedding_dim)
    deps = make_deps(conn, settings)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as cp:
        app = build_graph(deps, cp)
        for _ in range(40):
            assert (await run_turn(app, ALICE, "t1", "/help")).outcome == "help"
    conn.close()
