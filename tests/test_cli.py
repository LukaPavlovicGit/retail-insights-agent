from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from ria import db
from ria.bootstrap import chat_loop, parse_confirmation, run_init
from ria.cli import app as cli
from ria.golden.embed import FakeEmbedder
from ria.golden.loader import Trio
from ria.golden.store import TrioStore
from ria.graph.run import TurnResult
from ria.observability.trace import TraceStore, TurnTrace

SEED_DIR = Path(__file__).resolve().parents[1] / "data"


def seed_copy(tmp_path: Path) -> Path:
    """Copy the seed users and Trios into a temp data dir so init never writes into the repo's data/."""
    data = tmp_path / "data"
    shutil.copytree(SEED_DIR / "golden_trios", data / "golden_trios")
    shutil.copy(SEED_DIR / "users.yaml", data / "users.yaml")
    return data


def test_parse_confirmation():
    assert parse_confirmation("yes") and parse_confirmation(" Yes ") and parse_confirmation("y")
    assert not parse_confirmation("yes please") and not parse_confirmation("no") and not parse_confirmation("")


def test_run_init_loads_users_trios_and_embeddings(settings, tmp_path):
    settings = replace(settings, data_dir=seed_copy(tmp_path), db_path=tmp_path / "ria.db")
    conn = db.connect(settings.db_path)
    db.init_schema(conn, settings.embedding_dim)
    report = run_init(settings, conn, FakeEmbedder(settings.embedding_dim),
                      schema_fetch=lambda: {"users": {"id": "INT64"}, "orders": {}, "order_items": {}, "products": {}},
                      live_scope_values=lambda: ["Levi's", "Carhartt", "Dockers", "Wrangler", "Columbia", "Volcom", "Quiksilver", "Speedo"])
    assert report["users"] == 3 and report["trios"] == 15 and report["embedded"] == 15
    assert report["schema_source"] == "live" and report["unknown_scope_values"] == []
    assert TrioStore(conn).missing_embeddings() == []
    assert (tmp_path / "ria.db").exists()
    assert settings.schema_path.exists() and not (SEED_DIR / "schema.json").exists()
    conn.close()


def test_run_init_reports_unknown_scope_values_and_static_schema(settings, tmp_path):
    settings = replace(settings, data_dir=seed_copy(tmp_path), db_path=tmp_path / "ria.db")
    conn = db.connect(settings.db_path)
    db.init_schema(conn, settings.embedding_dim)
    report = run_init(settings, conn, FakeEmbedder(settings.embedding_dim), schema_fetch=None, live_scope_values=lambda: ["Levi's"])
    assert report["schema_source"] == "static" and "Carhartt" in report["unknown_scope_values"]
    assert report["scope_check"] == "ok"

    def boom():
        raise RuntimeError("bq down")

    failed = run_init(settings, conn, FakeEmbedder(settings.embedding_dim), schema_fetch=None, live_scope_values=boom)
    assert failed["scope_check"] == "failed" and failed["unknown_scope_values"] == []
    conn.close()


async def test_chat_loop_handles_confirmation_and_eof():
    outputs = []
    prompts = iter(["delete my reports", "yes", "hello"])
    calls = []

    class App:
        pass

    async def fake_run_turn(app, user, thread_id, text, resume=None, **kw):
        calls.append((text, resume))
        if resume is not None:
            return TurnResult("t1", "Deleted 1 report: X.", None, "deleted")
        if text.startswith("delete"):
            return TurnResult("t1", None, {"action": "delete_reports", "reports": [{"id": 1, "title": "X", "created_at": "d"}]}, "needs_confirmation")
        return TurnResult("t2", "Hi.", None, "ok")

    async def prompt_fn(p):
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError

    await chat_loop(App(), {"id": "alice", "name": "A", "role": "r", "scope": None}, "thread", prompt_fn, outputs.append, run_turn_fn=fake_run_turn)
    joined = "\n".join(str(o) for o in outputs)
    assert "X" in joined and "Deleted 1 report" in joined and "Hi." in joined
    assert "[turn t1 · deleted]" in outputs
    assert calls == [("delete my reports", None), ("", {"confirmed": True}), ("hello", None)]


async def test_chat_loop_survives_interrupt_during_turn():
    outputs = []
    prompts = iter(["slow", "again"])

    async def run_turn_fn(app, user, thread_id, text, resume=None, **kw):
        if text == "slow":
            raise KeyboardInterrupt
        return TurnResult("t3", "Fine.", None, "ok")

    async def prompt_fn(p):
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError

    await chat_loop(object(), {"id": "a", "name": "A", "role": "r", "scope": None}, "th", prompt_fn, outputs.append, run_turn_fn=run_turn_fn)
    assert "Turn cancelled." in outputs and "Fine." in outputs
    assert "/help" in outputs[0]


def test_stats_shows_rating_and_correction_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("RIA_DB_PATH", str(tmp_path / "ria.db"))
    monkeypatch.setenv("RIA_DATA_DIR", str(tmp_path))
    conn = db.connect(tmp_path / "ria.db")
    db.init_schema(conn, 768)
    TraceStore(conn).save(TurnTrace("r1", "t", "alice", "analysis", "ok", "2026-10-16T00:00:00+00:00", 120, rating=1, corrected=True))
    conn.close()
    out = CliRunner().invoke(cli, ["stats"]).output
    assert "rated good/bad" in out and "1/0" in out and "correction_rate" in out and "100%" in out


def test_trace_stats_and_golden_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("RIA_DB_PATH", str(tmp_path / "ria.db"))
    monkeypatch.setenv("RIA_DATA_DIR", str(tmp_path))
    conn = db.connect(tmp_path / "ria.db")
    db.init_schema(conn, 768)
    TraceStore(conn).save(TurnTrace("abc", "t", "alice", "analysis", "ok", "2026-10-16T00:00:00+00:00", 120,
                                    llm_calls=[{"node": "router", "model": "m", "tokens_in": 1, "tokens_out": 1, "latency_ms": 5, "fallback": False}]))
    tid = TrioStore(conn).insert(Trio(id=None, question="pending q", sql="SELECT 1", report="r", status="pending", source="user"))
    conn.close()
    runner = CliRunner()
    assert "router" in runner.invoke(cli, ["trace", "abc"]).output
    assert "turns" in runner.invoke(cli, ["stats"]).output
    listed = runner.invoke(cli, ["golden", "list"]).output
    assert "pending q" in listed and "[pending]" in listed
    assert runner.invoke(cli, ["golden", "promote", str(tid)]).exit_code == 0
    assert "pending q" not in runner.invoke(cli, ["golden", "list", "--status", "pending"]).output
    assert "not found" in runner.invoke(cli, ["trace", "nope"]).output
