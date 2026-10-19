from __future__ import annotations

import json
import os

import pytest

from ria import db
from ria.config import Settings
from ria.mcp_server import server
from ria.tools import FakeToolBox, MCPToolBox, open_mcp_toolbox


@pytest.fixture
def served_db(tmp_path, monkeypatch):
    path = tmp_path / "ria.db"
    c = db.connect(path)
    db.init_schema(c, 32)
    c.close()
    monkeypatch.setenv("RIA_DB_PATH", str(path))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    # Make google.auth.default() find nothing, so tests never reach Google even with ADC configured.
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing-creds.json"))
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path / "no-gcloud"))
    monkeypatch.setenv("GCE_METADATA_HOST", "127.0.0.1:1")
    server.reset()
    yield path
    server.reset()


def test_report_tools_direct(served_db):
    out = json.loads(server.reports_save("alice", "t1", "Title", "q", "SELECT 1", "body", ["x"]))
    assert out["ok"] and out["id"] == 1
    found = json.loads(server.reports_search("alice", text="title"))
    assert [r["id"] for r in found["reports"]] == [1] and found["reports"][0]["title"] == "Title"
    assert json.loads(server.reports_search("bob"))["reports"] == []
    assert json.loads(server.reports_delete("alice", [1]))["deleted"] == 1


def test_bq_schema_falls_back_to_static_without_credentials(served_db):
    out = json.loads(server.bq_schema())
    assert out["source"] == "static" and "order_items" in out["schema"]


def test_bq_query_without_credentials_is_graceful(served_db, monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(served_db.parent / "nope.json"))
    out = json.loads(server.bq_query("SELECT 1"))
    assert out["ok"] is False and out["error_kind"] in ("unknown", "circuit_open") and out["error"]


async def test_stdio_roundtrip_through_mcp(served_db):
    settings = Settings(db_path=served_db, gcp_project="test-project")
    async with open_mcp_toolbox(settings) as tb:
        assert isinstance(tb, MCPToolBox)
        saved = await tb.reports_save("alice", "t1", "Round trip", "q", "SELECT 1", "body", [])
        assert saved["ok"] and saved["id"] == 1
        found = await tb.reports_search("alice", text="round")
        assert [r["id"] for r in found["reports"]] == [1]
        deleted = await tb.reports_delete("alice", [1])
        assert deleted["deleted"] == 1
        schema = await tb.bq_schema()
        assert schema["source"] == "static" and "products" in schema["schema"]


async def test_fake_toolbox_defaults_and_queue():
    tb = FakeToolBox({"bq_query": [{"ok": False, "error": "boom", "error_kind": "syntax", "rows": [], "row_count": 0, "bytes": 0}]})
    first = await tb.bq_query("SELECT bad")
    assert first["ok"] is False
    second = await tb.bq_query("SELECT good")
    assert second["ok"] is True and second["rows"] == [{"value": 1}]
    assert (await tb.bq_dry_run("x"))["ok"] is True
    assert tb.calls[0] == ("bq_query", {"sql": "SELECT bad"})
