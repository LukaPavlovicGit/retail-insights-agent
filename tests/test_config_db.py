from __future__ import annotations

from pathlib import Path

from ria import db
from ria.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.primary_model == "gemini-3.8-flash"
    assert s.fallback_model == "gemini-3.5-flash-lite"
    assert s.embedding_model == "gemini-embedding-001"
    assert s.embedding_dim == 768
    assert s.dataset == "bigquery-public-data.thelook_ecommerce"
    assert s.max_repairs == 2
    assert s.max_llm_calls_per_turn == 8
    assert s.row_cap == 200
    assert s.scope_column == "brand" and s.min_group_size == 10
    assert s.definitions_path == Path("data") / "definitions.md"


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("RIA_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("RIA_BQ_MAX_BYTES", "123")
    monkeypatch.setenv("RIA_SCOPE_COLUMN", "category")
    monkeypatch.setenv("RIA_MIN_GROUP_SIZE", "25")
    s = Settings.from_env(env_file=None)
    assert s.scope_column == "category" and s.min_group_size == 25
    assert s.gemini_api_key == "k"
    assert s.gcp_project == "p"
    assert s.db_path == tmp_path / "x.db"
    assert s.bq_max_bytes == 123


def test_init_schema_creates_tables(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")}
    for t in ["users", "user_prefs", "reports", "trios", "traces", "llm_ledger", "trio_vec"]:
        assert t in names, t


def test_init_schema_is_idempotent(conn, settings):
    db.init_schema(conn, settings.embedding_dim)
    db.init_schema(conn, settings.embedding_dim)


def test_load_users_and_get_user(conn, tmp_path: Path):
    p = tmp_path / "users.yaml"
    p.write_text(
        "- id: alice\n  name: Alice Chen\n  role: VP Apparel\n  brands: [\"Levi's\", Carhartt]\n"
        "- id: carol\n  name: Carol Diaz\n  role: CEO\n  brands: ['*']\n"
    )
    users = db.load_users(conn, p)
    assert [u["id"] for u in users] == ["alice", "carol"]
    assert db.get_user(conn, "alice")["scope"] == ["Levi's", "Carhartt"]
    assert db.get_user(conn, "carol")["scope"] is None
    assert db.get_user(conn, "nobody") is None
