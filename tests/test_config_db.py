from __future__ import annotations

from pathlib import Path

import pytest

from ria import db
from ria.config import OrgDefaults, OrgDefaultsInvalid, Settings, render_definitions


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
    assert s.scope_column == "brand"
    assert s.org_defaults_path == Path("data") / "org_defaults.yaml"
    assert not hasattr(s, "min_group_size") and not hasattr(s, "definitions_path")


def test_checkpoint_db_is_a_separate_file_next_to_the_app_db():
    """The LangGraph checkpointer must not share a file with the app connection (see the integration test on deadlocks)."""
    s = Settings(db_path=Path("/x/ria.db"))
    assert s.checkpoint_db_path == Path("/x/ria-checkpoints.db") and s.checkpoint_db_path != s.db_path


def test_org_defaults_load_and_fallback(tmp_path):
    assert OrgDefaults.load(tmp_path / "missing.yaml") == OrgDefaults()
    f = tmp_path / "org_defaults.yaml"
    f.write_text("churn_window_days: 30\nrevenue_statuses: [Complete, Shipped]\nmin_group_size: 25\n")
    d = OrgDefaults.load(f)
    assert d == OrgDefaults(churn_window_days=30, revenue_statuses=("Complete", "Shipped"), min_group_size=25)
    f.write_text("min_group_size: 5\n")
    assert OrgDefaults.load(f) == OrgDefaults(min_group_size=5)
    f.write_text("")
    assert OrgDefaults.load(f) == OrgDefaults()
    p = Path(__file__).resolve().parents[1] / "data" / "org_defaults.yaml"
    assert p.exists()
    assert OrgDefaults.load(p) == OrgDefaults()


@pytest.mark.parametrize("text,key", [
    ("churn_window_days: 0", "churn_window_days"),
    ("churn_window_days: 3651", "churn_window_days"),
    ("churn_window_days: ninety", "churn_window_days"),
    ("churn_window_days: true", "churn_window_days"),
    ("revenue_statuses: []", "revenue_statuses"),
    ("revenue_statuses: Complete", "revenue_statuses"),
    ("revenue_statuses: [Complete, Lost]", "revenue_statuses"),
    ("min_group_size: 0", "min_group_size"),
    ("min_group_size: 1001", "min_group_size"),
    ("min_group_size: 2.5", "min_group_size"),
    ("bogus_key: 1", "bogus_key"),
    ("revenue_statuses: [Complete, Complete]", "revenue_statuses"),
    ("- Complete", "mapping"),
    ('churn_window_days: "90"', "churn_window_days"),
])
def test_org_defaults_validation_names_key(tmp_path, text, key):
    f = tmp_path / "org_defaults.yaml"
    f.write_text(text + "\n")
    with pytest.raises(OrgDefaultsInvalid, match=key):
        OrgDefaults.load(f)


def test_org_defaults_yaml_syntax_error_names_file(tmp_path):
    f = tmp_path / "org_defaults.yaml"
    f.write_text("revenue_statuses: [Complete\n")
    with pytest.raises(ValueError, match="org_defaults.yaml"):
        OrgDefaults.load(f)


def test_render_definitions():
    text = render_definitions(OrgDefaults())
    assert "90 days" in text and "Complete, Shipped or Processing" in text
    assert text.startswith("# Business definitions (org-wide defaults; read on every answer)")
    assert text.endswith("use theirs for that answer and say so.")
    custom = render_definitions(OrgDefaults(churn_window_days=30, revenue_statuses=("Complete", "Shipped")))
    assert "30 days" in custom and "status Complete or Shipped." in custom
    assert "status Complete." in render_definitions(OrgDefaults(revenue_statuses=("Complete",)))


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("RIA_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("RIA_BQ_MAX_BYTES", "123")
    monkeypatch.setenv("RIA_SCOPE_COLUMN", "category")
    s = Settings.from_env(env_file=None)
    assert s.scope_column == "category"
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
