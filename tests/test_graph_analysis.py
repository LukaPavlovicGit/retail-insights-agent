from __future__ import annotations

from ria.golden.loader import Trio
from ria.graph.analysis import (
    SqlPlan, make_analyze, make_execute_sql, make_generate_sql, make_give_up, make_guard_sql, make_render,
    make_repair_sql, make_retrieve_trios, route_after_execute, route_after_guard, rows_to_markdown,
)
from ria.tools import FakeToolBox
from tests.graph_fixtures import CAROL, make_deps, turn_state

DS = "bigquery-public-data.thelook_ecommerce"
GOOD_SQL = (
    f"SELECT p.category, SUM(oi.sale_price) AS revenue FROM `{DS}.order_items` oi "
    f"JOIN `{DS}.products` p ON p.id = oi.product_id GROUP BY 1 ORDER BY 2 DESC"
)
BRAND_IN = "IN ('Levi\\'s', 'Carhartt', 'Dockers', 'Wrangler')"
PII_SQL = f"SELECT email FROM `{DS}.users`"
ROWS = [{"category": "Jeans", "revenue": 1200.5}, {"category": "Pants", "revenue": 800.0}]


async def seed(deps, questions):
    for q in questions:
        i = deps.trio_store.insert(Trio(id=None, question=q, sql="SELECT 1", report=f"Report for {q}", tags=["all"]))
        deps.trio_store.set_embedding(i, await deps.embedder.embed(q, kind="document"))


async def test_retrieve_trios_failure_continues_without_examples(conn, settings):
    class BoomEmbedder:
        dim = 32

        async def embed(self, text, kind="query"):
            raise RuntimeError("429")

    deps = make_deps(conn, settings)
    deps.embedder = BoomEmbedder()
    out = await make_retrieve_trios(deps)(turn_state("monthly revenue?", guard_events=["x"]))
    assert out["retrieved_trios"] == [] and out["guard_events"] == ["x", "retrieval:failed:RuntimeError"]


async def test_retrieve_trios_node(conn, settings):
    deps = make_deps(conn, settings)
    await seed(deps, ["monthly revenue trend", "top customers by spend"])
    out = await make_retrieve_trios(deps)(turn_state("what is the monthly revenue trend?"))
    assert out["retrieved_trios"][0]["question"] == "monthly revenue trend"


async def test_generate_sql_strips_fences_and_uses_examples(conn, settings):
    deps = make_deps(conn, settings, {"generate_sql": [SqlPlan(plan="p", sql="```sql\nSELECT 1\n```")]})
    state = turn_state("q", retrieved_trios=[{"question": "monthly revenue", "sql": "SELECT 2", "report": "R"}])
    out = await make_generate_sql(deps)(state)
    assert out["sql"] == "SELECT 1" and len(out["llm_calls"]) == 1
    system = deps.llm.calls[0]["system"]
    assert "monthly revenue" in system and "SELECT 2" in system and "Levi's, Carhartt" in system and "order_items" in system


async def test_guard_node_rewrites_and_dry_runs(conn, settings):
    tb = FakeToolBox()
    deps = make_deps(conn, settings, toolbox=tb)
    out = await make_guard_sql(deps)(turn_state("q", sql=GOOD_SQL))
    assert BRAND_IN in out["sql"] and out["sql_error"] is None
    assert "scope:injected" in out["guard_events"] and "dryrun:ok" in out["guard_events"]
    assert tb.calls[0][0] == "bq_dry_run" and BRAND_IN in tb.calls[0][1]["sql"]
    assert route_after_guard(out) == "execute_sql"


async def test_guard_node_rejects_pii_and_routes_to_repair_then_give_up(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_guard_sql(deps)(turn_state("q", sql=PII_SQL))
    assert out["sql_error_kind"] == "guard" and "email" in out["sql_error"]
    assert out["sql_attempts"][-1]["kind"] == "guard" and "pii:rejected" in out["guard_events"]
    assert route_after_guard({**out, "repairs": 0}) == "repair_sql"
    assert route_after_guard({**out, "repairs": settings.max_repairs}) == "give_up"


async def test_guard_node_dry_run_bytes_cap_gives_up(conn, settings):
    tb = FakeToolBox({"bq_dry_run": [{"ok": False, "bytes": 5_000_000_000, "error": "Query would scan 5,000,000,000 bytes", "error_kind": "bytes_cap"}]})
    deps = make_deps(conn, settings, toolbox=tb)
    out = await make_guard_sql(deps)(turn_state("q", sql=GOOD_SQL))
    assert out["sql_error_kind"] == "bytes_cap" and route_after_guard(out) == "give_up"


async def test_execute_sql_success_empty_and_circuit(conn, settings):
    tb = FakeToolBox({"bq_query": [
        {"ok": True, "rows": ROWS, "row_count": 2, "bytes": 99, "error": None, "error_kind": None},
        {"ok": True, "rows": [], "row_count": 0, "bytes": 1, "error": None, "error_kind": None},
        {"ok": True, "rows": [], "row_count": 0, "bytes": 1, "error": None, "error_kind": None},
        {"ok": False, "rows": [], "row_count": 0, "bytes": 0, "error": "down", "error_kind": "circuit_open"},
    ]})
    deps = make_deps(conn, settings, toolbox=tb)
    node = make_execute_sql(deps)
    ok = await node(turn_state("q", sql="SELECT 1"))
    assert ok["result_rows"] == ROWS and ok["result_bytes"] == 99 and route_after_execute(ok) == "analyze"
    empty1 = await node(turn_state("q", sql="SELECT 1"))
    assert empty1["sql_error_kind"] == "empty" and route_after_execute(empty1) == "repair_sql"
    empty2 = await node(turn_state("q", sql="SELECT 1", sql_attempts=empty1["sql_attempts"]))
    assert route_after_execute(empty2) == "give_up"
    down = await node(turn_state("q", sql="SELECT 1"))
    assert down["sql_error_kind"] == "circuit_open" and route_after_execute(down) == "give_up"


async def test_execute_syntax_error_routes_to_repair(conn, settings):
    tb = FakeToolBox({"bq_query": [{"ok": False, "rows": [], "row_count": 0, "bytes": 0, "error": "Syntax error", "error_kind": "syntax"}]})
    deps = make_deps(conn, settings, toolbox=tb)
    out = await make_execute_sql(deps)(turn_state("q", sql="SELECT bad"))
    assert route_after_execute(out) == "repair_sql" and out["sql_attempts"][-1]["error"] == "Syntax error"


async def test_repair_sql_includes_history_and_increments(conn, settings):
    deps = make_deps(conn, settings, {"repair_sql": [SqlPlan(plan="fix", sql="SELECT 2")]})
    attempts = [{"sql": "SELECT bad", "error": "Syntax error", "kind": "syntax", "row_count": 0, "bytes": 0}]
    out = await make_repair_sql(deps)(turn_state("q", sql="SELECT bad", sql_attempts=attempts, repairs=0, sql_error="Syntax error", sql_error_kind="syntax"))
    assert out["sql"] == "SELECT 2" and out["repairs"] == 1 and out["sql_error"] is None
    assert "SELECT bad" in deps.llm.calls[0]["system"] and "Syntax error" in deps.llm.calls[0]["system"]


async def test_give_up_messages(conn, settings):
    deps = make_deps(conn, settings)
    node = make_give_up(deps)
    empty = await node(turn_state("q", sql="SELECT 1", sql_error_kind="empty", sql_attempts=[{"sql": "SELECT 1", "error": "no rows", "kind": "empty", "row_count": 0, "bytes": 0}]))
    assert empty["outcome"] == "no_data" and "Levi's, Carhartt, Dockers, Wrangler" in empty["rendered"] and "brand" in empty["rendered"]
    unscoped = await node(turn_state("q", user=CAROL, sql="SELECT 1", sql_error_kind="empty", sql_attempts=[]))
    assert unscoped["outcome"] == "no_data" and "Your access covers" not in unscoped["rendered"] and "SELECT 1" in unscoped["rendered"]
    down = await node(turn_state("q", sql_error_kind="circuit_open", sql_attempts=[]))
    assert down["outcome"] == "error" and "not responding" in down["rendered"]
    big = await node(turn_state("q", sql_error_kind="bytes_cap", sql_error="Query would scan 5,000,000,000 bytes", sql_attempts=[]))
    assert big["outcome"] == "gave_up" and "5,000,000,000" in big["rendered"]
    gen = await node(turn_state("q", sql_error_kind="syntax", sql_attempts=[{"sql": "A", "error": "e1", "kind": "syntax", "row_count": 0, "bytes": 0}, {"sql": "B", "error": "e2", "kind": "guard", "row_count": 0, "bytes": 0}]))
    assert gen["outcome"] == "gave_up" and "e1" in gen["rendered"] and "e2" in gen["rendered"] and "2 attempts" in gen["rendered"]


async def test_analyze_uses_prefs_and_rows(conn, settings):
    deps = make_deps(conn, settings, {"analyze": ["Jeans lead revenue."]})
    deps.prefs_store.apply("alice", {"format": "bullets"})
    state = turn_state("which category leads?", result_rows=ROWS, sql="SELECT 1",
                       retrieved_trios=[{"question": "x", "sql": "s", "report": "Style ref text"}])
    out = await make_analyze(deps)(state)
    assert out["analysis"] == "Jeans lead revenue."
    call = deps.llm.calls[0]
    assert "bullet points" in call["system"] and "Style ref text" in call["system"]
    assert "Jeans" in call["messages"][-1].content and "1200.5" in call["messages"][-1].content


async def test_prompts_carry_business_definitions(conn, settings):
    deps = make_deps(conn, settings, {
        "generate_sql": [SqlPlan(plan="p", sql="SELECT 1")], "repair_sql": [SqlPlan(plan="p", sql="SELECT 2")],
        "analyze": ["ok"],
    })
    await make_generate_sql(deps)(turn_state("q"))
    await make_repair_sql(deps)(turn_state("q", sql_attempts=[], repairs=0))
    await make_analyze(deps)(turn_state("q", result_rows=ROWS, sql="SELECT 1"))
    for call in deps.llm.calls:
        assert "90 days" in call["system"] and "Complete, Shipped or Processing" in call["system"]
    assert "state the definition" in deps.llm.calls[2]["system"]
    layering = "user's saved preferences only change how the answer is presented, never a definition"
    assert layering in deps.llm.calls[2]["system"]
    assert "revenue counts items with status Complete, Shipped or Processing" in deps.llm.calls[0]["system"]


async def test_guard_node_uses_org_default_group_size(conn, settings):
    settings.data_dir.mkdir(parents=True)
    (settings.org_defaults_path).write_text("min_group_size: 25\nchurn_window_days: 30\nrevenue_statuses: [Complete]\n")
    deps = make_deps(conn, settings, {"generate_sql": [SqlPlan(plan="p", sql="SELECT 1")]})
    sql = (f"SELECT u.gender, COUNT(*) AS n FROM `{DS}.users` u JOIN `{DS}.order_items` oi ON oi.user_id = u.id "
           f"JOIN `{DS}.products` p ON p.id = oi.product_id GROUP BY u.gender")
    out = await make_guard_sql(deps)(turn_state("q", sql=sql))
    assert ">= 25" in out["sql"]
    await make_generate_sql(deps)(turn_state("q"))
    system = deps.llm.calls[0]["system"]
    assert "30 days" in system and "revenue counts items with status Complete)" in system


async def test_render_table_and_deep(conn, settings):
    deps = make_deps(conn, settings)
    deps.prefs_store.apply("alice", {"format": "table", "depth_up": True})
    out = await make_render(deps)(turn_state("q", analysis="Jeans lead.", result_rows=ROWS, sql="SELECT 1"))
    assert out["outcome"] == "ok" and out["rendered"].startswith("Jeans lead.")
    assert "| category | revenue |" in out["rendered"] and "```sql" in out["rendered"]


async def test_render_prose_has_no_table(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_render(deps)(turn_state("q", analysis="Jeans lead.", result_rows=ROWS, sql="SELECT 1"))
    assert "| category" not in out["rendered"] and "```sql" not in out["rendered"]


def test_rows_to_markdown_limits_rows():
    rows = [{"a": i, "b": f"x{i}"} for i in range(30)]
    md = rows_to_markdown(rows, limit=5)
    assert md.count("\n") == 6 + 1 and "(showing 5 of 30 rows)" in md
