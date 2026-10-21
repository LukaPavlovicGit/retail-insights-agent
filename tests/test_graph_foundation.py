from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from ria.graph.lifecycle import make_begin_turn, make_fail, make_finalize, make_update_memory, make_write_trace
from ria.graph.nodes_common import BudgetExceeded, after, call_llm, route_intent, scope_text, timed
from ria.graph.router import RouteDecision, make_ack_preference, make_refuse, make_router
from ria.graph.prerouter import make_pre_router
from ria.graph.router import make_help, make_rating
from ria.graph.schema_qa import make_schema_answer, make_schema_overview
from ria.llm.client import LLMUnavailable
from tests.graph_fixtures import CAROL, make_deps, turn_state


async def test_timed_records_timing_and_converts_llm_errors():
    async def ok(state):
        return {"x": 1}

    async def boom(state):
        raise LLMUnavailable("down")

    out = await timed("ok", ok)({"node_timings": []})
    assert out["x"] == 1 and out["node_timings"][0]["node"] == "ok"
    err = await timed("boom", boom)({"node_timings": []})
    assert err["outcome"] == "error" and "LLMUnavailable" in err["error"]


async def test_call_llm_enforces_budget(conn, settings):
    deps = make_deps(conn, settings)
    state = turn_state("q", llm_calls=[{} for _ in range(settings.max_llm_calls_per_turn)])
    with pytest.raises(BudgetExceeded):
        await call_llm(deps, state, node="x", system="s", messages=[])


def test_routing_helpers():
    assert route_intent({"intent": "analysis"}) == "analysis"
    assert route_intent({"intent": "analysis", "error": "x"}) == "fail"
    assert after("next")({}) == "next" and after("next")({"error": "x"}) == "fail"


def test_scope_text_pluralises_column():
    assert scope_text(turn_state("q"), "brand").startswith("only these brands: Levi's")
    assert scope_text(turn_state("q", user=CAROL), "brand") == "all brands"
    assert scope_text(turn_state("q", user=CAROL), "category") == "all categories"
    assert scope_text(turn_state("q", user=CAROL), "region") == "all regions"


async def test_router_uses_llm_decision(conn, settings):
    deps = make_deps(conn, settings, {"router": [RouteDecision(intent="report_delete", reason="r", report_filter_scope="text", report_filter_text="Client X")]})
    out = await make_router(deps)(turn_state("delete all reports mentioning Client X"))
    assert out["intent"] == "report_delete" and out["report_filter_text"] == "Client X" and len(out["llm_calls"]) == 1
    system = deps.llm.calls[0]["system"]
    assert "Alice Chen" in system and "only these brands: Levi's, Carhartt, Dockers, Wrangler" in system


async def test_router_empty_message_is_off_topic_without_llm(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_router(deps)(turn_state("   "))
    assert out["intent"] == "off_topic" and deps.llm.calls == []


async def test_refuse_and_ack_preference(conn, settings):
    deps = make_deps(conn, settings)
    r = await make_refuse(deps)(turn_state("write me a poem"))
    assert r["outcome"] == "refused" and "sales" in r["rendered"]
    a = await make_ack_preference(deps)(turn_state("please use tables from now on"))
    assert a["outcome"] == "ok" and "Format: table" in a["rendered"]
    assert deps.prefs_store.get("alice").format == "table"


async def test_schema_answer(conn, settings):
    deps = make_deps(conn, settings, {"schema_answer": ["Four tables..."]})
    out = await make_schema_answer(deps)(turn_state("what data do we have?"))
    assert out["rendered"] == "Four tables..." and out["outcome"] == "ok"
    assert "order_items" in deps.llm.calls[0]["system"]


async def test_pre_router_sets_intent_for_commands_only(conn, settings):
    node = make_pre_router(make_deps(conn, settings))
    assert await node(turn_state("how are jeans doing?")) == {}
    out = await node(turn_state("/reports"))
    assert out["intent"] == "report_list" and out["command"]["kind"] == "reports"
    out = await node(turn_state("/delete mentioning Acme"))
    assert out["intent"] == "report_delete" and (out["report_filter_scope"], out["report_filter_text"]) == ("text", "Acme")
    assert (await node(turn_state("/report 4")))["intent"] == "report_show"
    assert (await node(turn_state("/schema")))["intent"] == "schema_overview"
    assert (await node(turn_state("/bad no")))["intent"] == "rating" and (await node(turn_state("/help")))["intent"] == "help"


async def test_rating_node(conn, settings):
    from ria.observability.trace import TurnTrace
    deps = make_deps(conn, settings)
    deps.trace_store.save(TurnTrace(turn_id="prev", thread_id="t1", user_id="alice", intent="analysis", outcome="ok",
                                    started_at="2026-10-16T10:00:00+00:00", duration_ms=1))
    node = make_rating(deps)
    none = await node(turn_state("/good", command={"kind": "rating", "rating": 1, "note": ""}))
    assert none["rendered"] == "Nothing to rate yet. Ask a question first." and none["outcome"] == "ok"
    out = await node(turn_state("/bad slow", last_turn_id="prev", command={"kind": "rating", "rating": -1, "note": "slow"}))
    assert out["outcome"] == "rated" and "thumbs down" in out["rendered"]
    t = deps.trace_store.get("prev")
    assert (t.rating, t.rating_note) == (-1, "slow") and deps.llm.calls == []


async def test_help_and_schema_overview_make_no_model_call(conn, settings):
    deps = make_deps(conn, settings)
    h = await make_help(deps)(turn_state("/help"))
    assert h["outcome"] == "help" and "/reports" in h["rendered"]
    s = await make_schema_overview(deps)(turn_state("/schema"))
    assert s["outcome"] == "ok" and "order_items" in s["rendered"] and deps.llm.calls == []


async def test_begin_turn_resets_and_summarizes(conn, settings):
    deps = make_deps(conn, settings, {"summarize": ["Earlier: asked about jeans."]})
    msgs = []
    for i in range(10):
        msgs.append(HumanMessage(content=f"q{i}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i}", id=f"a{i}"))
    state = turn_state("q10", sql="OLD", llm_calls=[{"node": "old"}])
    state["messages"] = msgs + [HumanMessage(content="q10", id="h10")]
    out = await make_begin_turn(deps)(state)
    assert out["sql"] is None and out["question"] == "q10"
    assert out["summary"] == "Earlier: asked about jeans."
    removed = [m for m in out["messages"] if isinstance(m, RemoveMessage)]
    assert len(removed) == 21 - 2 * settings.window_turns
    assert out["llm_calls"][0]["node"] == "summarize"


async def test_begin_turn_skips_summary_when_model_down(conn, settings):
    deps = make_deps(conn, settings, {"summarize": [LLMUnavailable("down")]})
    state = turn_state("/reports")
    state["messages"] = [HumanMessage(content=f"q{i}", id=f"h{i}") for i in range(30)]
    out = await make_begin_turn(deps)(state)
    assert "summary" not in out and "messages" not in out and out["guard_events"] == ["summary:skipped_llm_unavailable"]


async def test_finalize_last_turn_id_only_for_answer_outcomes(conn, settings):
    deps = make_deps(conn, settings)
    fin = make_finalize(deps)
    for outcome in ("ok", "no_data", "gave_up", "refused", "error", "saved", "deleted", "cancelled"):
        assert (await fin(turn_state("q", rendered="x", outcome=outcome, turn_id="T")))["last_turn_id"] == "T"
    for outcome in ("rated", "help", "listed", "shown"):
        assert "last_turn_id" not in await fin(turn_state("q", rendered="x", outcome=outcome, turn_id="T"))
    assert "last_turn_id" not in await fin(turn_state("q", rendered="x", outcome="ok", intent="rating"))
    assert "last_turn_id" not in await fin(turn_state("/schema", rendered="x", outcome="ok", intent="schema_overview"))


async def test_router_marks_previous_turn_corrected(conn, settings):
    from ria.observability.trace import TurnTrace
    deps = make_deps(conn, settings, {"router": [RouteDecision(intent="analysis", is_correction=True), RouteDecision(intent="analysis", is_correction=True)]})
    deps.trace_store.save(TurnTrace(turn_id="prev", thread_id="t1", user_id="alice", intent="analysis", outcome="ok",
                                    started_at="2026-10-16T10:00:00+00:00", duration_ms=1))
    first = await make_router(deps)(turn_state("no, I meant last year"))  # no last_turn_id yet
    assert first["correction_of"] is None and not deps.trace_store.get("prev").corrected
    out = await make_router(deps)(turn_state("no, I meant last year", last_turn_id="prev"))
    assert out["correction_of"] == "prev" and deps.trace_store.get("prev").corrected


async def test_begin_turn_short_history_no_summary(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_begin_turn(deps)(turn_state("hello"))
    assert "summary" not in out and deps.llm.calls == []


async def test_finalize_masks_and_appends_and_remembers_last(conn, settings):
    deps = make_deps(conn, settings)
    state = turn_state("q", intent="analysis", rendered="Email jane@x.com for details", analysis="A", sql="SELECT 1", outcome="ok")
    out = await make_finalize(deps)(state)
    assert out["rendered"] == "Email [email] for details"
    assert isinstance(out["messages"][0], AIMessage) and out["last_sql"] == "SELECT 1" and out["last_question"] == "q"


async def test_finalize_sets_last_turn_id(conn, settings):
    deps = make_deps(conn, settings)
    state = turn_state("q", intent="schema_question", rendered="hi")
    out = await make_finalize(deps)(state)
    assert out["last_turn_id"] == state["turn_id"]


async def test_finalize_without_rendered_is_error(conn, settings):
    deps = make_deps(conn, settings)
    out = await make_finalize(deps)(turn_state("q", intent="analysis"))
    assert out["outcome"] == "error" and "turn-1" in out["rendered"]


async def test_fail_messages(conn, settings):
    deps = make_deps(conn, settings)
    assert "budget" in (await make_fail(deps)(turn_state("q", error="BudgetExceeded: x")))["rendered"]
    assert "unavailable" in (await make_fail(deps)(turn_state("q", error="LLMUnavailable: x")))["rendered"]


async def test_update_memory_skips_schema_questions(conn, settings):
    deps = make_deps(conn, settings)
    await make_update_memory(deps)(turn_state("show me the tables", intent="schema_question"))
    assert deps.prefs_store.get("alice") == deps.prefs_store.get("nobody")  # unknown user = defaults


async def test_update_memory_and_write_trace(conn, settings):
    deps = make_deps(conn, settings)
    state = turn_state("show it as a table next time", intent="analysis", outcome="ok",
                       llm_calls=[{"node": "router", "model": "fake", "tokens_in": 1, "tokens_out": 1, "latency_ms": 1, "fallback": False}],
                       sql_attempts=[{"sql": "SELECT 1", "error": None, "row_count": 1, "bytes": 5}], result_bytes=5)
    await make_update_memory(deps)(state)
    assert deps.prefs_store.get("alice").format == "table"
    await make_write_trace(deps)(state)
    t = deps.trace_store.get("turn-1")
    assert t.intent == "analysis" and t.bq_bytes == 5 and t.tokens_in == 1
