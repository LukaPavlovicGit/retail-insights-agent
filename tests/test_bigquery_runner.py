from __future__ import annotations

import datetime as dt
from decimal import Decimal

from ria.mcp_server.bigquery import BigQueryRunner, CircuitBreaker, json_safe


class FakeJob:
    def __init__(self, rows=None, total_bytes=100, error=None, timeout=False):
        self._rows = rows or []
        self.total_bytes_processed = total_bytes
        self._error = error
        self._timeout = timeout

    def result(self, timeout=None, max_results=None):
        if self._timeout:
            import concurrent.futures
            raise concurrent.futures.TimeoutError("deadline")
        if self._error:
            raise self._error
        return self._rows[: max_results or None]


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.configs = []

    def query(self, sql, job_config=None):
        self.configs.append(job_config)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class BadRequest(Exception):
    code = 400


def test_json_safe_converts_bigquery_types():
    value = {
        "d": Decimal("1.50"), "day": dt.date(2026, 1, 2), "ts": dt.datetime(2026, 1, 2, 3, 4, 5),
        "nan": float("nan"), "b": b"x", "nested": [Decimal("2"), {"k": dt.date(2026, 1, 1)}],
    }
    out = json_safe(value)
    assert out["d"] == 1.5 and out["day"] == "2026-01-02" and out["ts"].startswith("2026-01-02T03:04:05")
    assert out["nan"] is None and out["b"] == "x" and out["nested"] == [2.0, {"k": "2026-01-01"}]


def test_breaker_opens_after_threshold_and_cools_down():
    t = [0.0]
    b = CircuitBreaker(threshold=3, cooldown_s=300, clock=lambda: t[0])
    b.record_failure(); b.record_failure()
    assert not b.is_open
    b.record_failure()
    assert b.is_open
    t[0] = 301
    assert not b.is_open
    b.record_success()
    assert b.failures == 0


def test_dry_run_reports_bytes_and_cap():
    client = FakeClient([FakeJob(total_bytes=500), FakeJob(total_bytes=5_000)])
    r = BigQueryRunner("p", "ds", max_bytes=1_000, timeout_s=5, row_cap=10, client=client)
    assert r.dry_run("SELECT 1") == {"ok": True, "bytes": 500, "error": None, "error_kind": None}
    capped = r.dry_run("SELECT 2")
    assert not capped["ok"] and capped["error_kind"] == "bytes_cap" and "5,000" in capped["error"]
    assert client.configs[0].dry_run is True


def test_query_returns_rows_and_sets_byte_cap():
    rows = [{"a": Decimal("1"), "b": dt.date(2026, 1, 1)}]
    client = FakeClient([FakeJob(rows=rows, total_bytes=42)])
    r = BigQueryRunner("p", "ds", max_bytes=1_000, timeout_s=5, row_cap=10, client=client)
    out = r.query("SELECT 1")
    assert out == {"ok": True, "rows": [{"a": 1.0, "b": "2026-01-01"}], "row_count": 1, "bytes": 42, "error": None, "error_kind": None}
    assert client.configs[0].maximum_bytes_billed == 1_000


def test_query_classifies_errors_and_trips_breaker():
    client = FakeClient([BadRequest("Syntax error: Unexpected keyword"), FakeJob(timeout=True), RuntimeError("quotaExceeded"), RuntimeError("x")])
    r = BigQueryRunner("p", "ds", max_bytes=1_000, timeout_s=5, row_cap=10, client=client, breaker=CircuitBreaker(threshold=3, cooldown_s=60))
    assert r.query("a")["error_kind"] == "syntax"
    assert r.query("b")["error_kind"] == "timeout"
    assert r.query("c")["error_kind"] == "quota"
    # syntax errors don't count toward the breaker; timeout + quota do, so one more failure opens it
    out = r.query("d")
    assert out["error_kind"] == "unknown"
    assert r.breaker.is_open
    assert r.query("e")["error_kind"] == "circuit_open" and len(client.outcomes) == 0


def test_classification_of_syntax_auth_and_bytes_billed_errors():
    client = FakeClient([
        BadRequest("Unrecognized name: foo at [1:8]"),
        RuntimeError("invalid_grant: Bad Request"),
        BadRequest("Query exceeded limit for bytes billed: 1000000000"),
    ])
    r = BigQueryRunner("p", "ds", max_bytes=1_000, timeout_s=5, row_cap=10, client=client)
    assert r.query("a")["error_kind"] == "syntax" and r.breaker.failures == 0
    assert r.query("b")["error_kind"] == "unknown" and r.breaker.failures == 1
    assert r.query("c")["error_kind"] == "bytes_cap" and r.breaker.failures == 1


def test_query_success_resets_breaker():
    client = FakeClient([RuntimeError("x"), FakeJob(rows=[])])
    r = BigQueryRunner("p", "ds", max_bytes=1_000, timeout_s=5, row_cap=10, client=client)
    r.query("a")
    assert r.breaker.failures == 1
    r.query("b")
    assert r.breaker.failures == 0
