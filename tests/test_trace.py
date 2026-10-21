from __future__ import annotations

import json

from rich.console import Console

from ria.observability.report import render_stats, render_trace
from ria.observability.trace import TraceStore, TurnTrace


def make(turn_id, outcome="ok", repairs=0, fallback=False, ms=100, intent="analysis", guard=()):
    attempts = [{"sql": "SELECT 1", "error": None, "row_count": 3, "bytes": 10}]
    for i in range(repairs):
        attempts.insert(0, {"sql": f"SELECT bad{i}", "error": "boom", "row_count": 0, "bytes": 0})
    return TurnTrace(
        turn_id=turn_id, thread_id="t", user_id="alice", intent=intent, outcome=outcome,
        started_at="2026-10-16T10:00:00+00:00", duration_ms=ms,
        node_timings=[{"node": "router", "ms": 20}, {"node": "analyze", "ms": 60}],
        llm_calls=[{"node": "router", "model": "m", "tokens_in": 10, "tokens_out": 5, "latency_ms": 20, "fallback": fallback}],
        sql_attempts=attempts, guard_events=list(guard), bq_bytes=10,
    )


def test_save_get_and_jsonl(conn, tmp_path):
    s = TraceStore(conn, jsonl_path=tmp_path / "traces.jsonl")
    s.save(make("a", repairs=1, guard=["pii:ok", "scope:injected"]))
    t = s.get("a")
    assert t.repairs == 1 and t.tokens_in == 10 and t.guard_events == ["pii:ok", "scope:injected"]
    lines = (tmp_path / "traces.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["turn_id"] == "a"
    assert s.get("missing") is None


def test_ledger_rows_written(conn):
    s = TraceStore(conn)
    s.save(make("a"))
    n = conn.execute("SELECT COUNT(*) FROM llm_ledger WHERE turn_id = 'a'").fetchone()[0]
    assert n == 1


def test_stats(conn):
    s = TraceStore(conn)
    s.save(make("1", ms=100))
    s.save(make("2", ms=300, repairs=2, fallback=True, guard=["pii:rejected"]))
    s.save(make("3", outcome="refused", intent="off_topic", ms=50))
    s.save(make("4", outcome="error", ms=500))
    st = s.stats()
    assert st["turns"] == 4
    assert st["success_rate"] == 0.5
    assert st["refusal_rate"] == 0.25
    assert st["error_rate"] == 0.25
    assert st["fallback_rate"] == 0.25
    assert abs(st["mean_repairs"] - 2 / 3) < 1e-9  # analysis-intent turns only: (0 + 2 + 0) / 3
    assert st["guard_rejections"] == {"pii": 1}
    # round-half-even index of pct*(n-1) over sorted 50,100,300,500: p50 -> idx 2 (300), p95 -> idx 3 (500)
    assert st["p50_ms"] == 300 and st["p95_ms"] == 500
    assert st["by_intent"] == {"analysis": 3, "off_topic": 1}


def test_set_rating_and_corrected_round_trip(conn):
    s = TraceStore(conn)
    s.save(make("a"))
    assert s.get("a").rating is None and s.get("a").rating_note == "" and s.get("a").corrected is False
    assert s.set_rating("a", -1, "wrong brand") is True
    assert s.set_corrected("a") is True
    t = s.get("a")
    assert t.rating == -1 and t.rating_note == "wrong brand" and t.corrected is True
    assert t.to_dict()["rating"] == -1 and t.to_dict()["corrected"] is True
    assert s.set_rating("nope", 1) is False
    assert s.set_corrected("nope") is False


def test_stats_ratings_and_correction_rate(conn):
    s = TraceStore(conn)
    for i in "123":
        s.save(make(i))
    s.save(make("4", outcome="listed", intent="list_reports"))
    s.set_rating("1", 1)
    s.set_rating("2", 1)
    s.set_rating("3", -1)
    s.set_corrected("3")
    st = s.stats()
    assert st["rated_good"] == 2 and st["rated_bad"] == 1
    assert abs(st["correction_rate"] - 1 / 3) < 1e-9
    assert st["success_rate"] == 1.0  # "listed" counts as success
    for o in ("shown", "rated", "help"):
        s.save(make(f"o-{o}", outcome=o, intent="x"))
    assert s.stats()["success_rate"] == 1.0


def test_correction_rate_zero_without_analysis_turns(conn):
    s = TraceStore(conn)
    s.save(make("a", intent="off_topic"))
    assert s.stats()["correction_rate"] == 0.0


def test_renderers_show_rating_and_corrected(conn):
    s = TraceStore(conn)
    s.save(make("a"))
    s.set_rating("a", 1)
    s.set_corrected("a")
    console = Console(record=True, width=100)
    console.print(render_trace(s.get("a")))
    console.print(render_stats(s.stats()))
    text = console.export_text()
    assert "rating" in text and "corrected" in text and "rated good/bad" in text and "correction_rate" in text


def test_stats_empty(conn):
    assert TraceStore(conn).stats()["turns"] == 0


def test_renderers_produce_text(conn):
    console = Console(record=True, width=100)
    console.print(render_trace(make("a", repairs=1)))
    console.print(render_stats(TraceStore(conn).stats()))
    text = console.export_text()
    assert "router" in text and "SELECT bad0" in text and "turns" in text


def test_render_trace_keeps_bracketed_text_literal(conn):
    t = make("a", fallback=True)
    t.sql_attempts[0]["error"] = "Syntax error [at 1:8]"
    t.sql_attempts[0]["sql"] = "SELECT '[/x]' AS s"
    console = Console(record=True, width=100)
    console.print(render_trace(t))
    text = console.export_text()
    assert "[fallback]" in text and "[at 1:8]" in text and "[/x]" in text
