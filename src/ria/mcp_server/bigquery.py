"""Guarded BigQuery access for the MCP server: dry runs, capped queries, error classification and a circuit breaker.

`BigQueryRunner` is the only BigQuery client in the project. It never raises on query failure; it returns a dict with an
`error_kind` that the graph uses to decide whether to repair the SQL or give up. Cost and size limits (bytes
scanned, timeout, row cap) are enforced here. Used by `mcp_server.server` and by the `init` and `doctor` CLI commands.
The Google client library is imported lazily, so importing this module needs no credentials.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import math
import time
from decimal import Decimal
from typing import Any

TABLES = ("orders", "order_items", "products", "users")


def json_safe(value: Any) -> Any:
    """Recursively convert BigQuery values to JSON-serialisable ones: Decimal to float, NaN and infinity to None,
    dates and times to ISO strings, bytes to text, and Row-like objects to dicts."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "items") and callable(value.items):  # bigquery Row
        return {str(k): json_safe(v) for k, v in value.items()}
    return value


class CircuitBreaker:
    """Stop calling a failing data source for a while. After `threshold` consecutive failures it opens for
    `cooldown_s`, during which `BigQueryRunner` answers immediately with `circuit_open`. When the cooldown ends it
    closes and the failure count restarts from zero (there is no single-probe half-open state). A success resets the
    count. `clock` is injectable for tests."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 300.0, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.failures = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        if self._open_until and self._clock() < self._open_until:
            return True
        if self._open_until and self._clock() >= self._open_until:
            self._open_until = 0.0
            self.failures = 0
        return False

    def record_success(self) -> None:
        self.failures = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self._open_until = self._clock() + self.cooldown_s


_SYNTAX_MARKERS = ("syntax error", "unrecognized name", "not found: table", "not found: dataset",
                   "no matching signature", "invalid query", "column", "function not found")
_NOT_SYNTAX_MARKERS = ("credential", "grant", "auth", "permission", "billed")
# Errors caused by the user's query, not the data source: they never trip the breaker.
_NOT_BREAKER_KINDS = ("syntax", "bytes_cap")


def classify_error(exc: Exception) -> str:
    """Map a BigQuery exception to `timeout`, `bytes_cap`, `quota`, `syntax` or `unknown`.

    `syntax` means the query itself is wrong (the LLM can fix it); it also covers a bare HTTP 400 unless the text
    points at credentials or permissions. `unknown`, `timeout` and `quota` suggest the data source is unhealthy.
    Classification is by message text, so it is a heuristic.
    """
    if isinstance(exc, (concurrent.futures.TimeoutError, TimeoutError)):
        return "timeout"
    text = f"{type(exc).__name__}: {exc}".lower()
    if "timeout" in text or "deadline" in text:
        return "timeout"
    if "bytes billed" in text or "maximumbytesbilled" in text:
        return "bytes_cap"
    if "quota" in text or "rate limit" in text or "ratelimitexceeded" in text:
        return "quota"
    if any(m in text for m in _SYNTAX_MARKERS):
        return "syntax"
    if getattr(exc, "code", None) == 400 and not any(m in text for m in _NOT_SYNTAX_MARKERS):
        return "syntax"
    return "unknown"


class BigQueryRunner:
    """Run SQL against one dataset with limits. `max_bytes` caps bytes scanned, `timeout_s` caps waiting, `row_cap`
    caps rows fetched. `client` and `breaker` can be injected for tests; otherwise a `bigquery.Client` is created on
    first use with the default Google credentials. All methods return plain dicts of the form
    `{ok, bytes, error, error_kind, ...}` and do not raise on query errors."""

    def __init__(self, project: str | None, dataset: str, max_bytes: int, timeout_s: int, row_cap: int,
                 client=None, breaker: CircuitBreaker | None = None):
        self.project = project
        self.dataset = dataset
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.row_cap = row_cap
        self._client = client
        self.breaker = breaker or CircuitBreaker()

    @property
    def client(self):
        if self._client is None:
            from google.cloud import bigquery

            self._client = bigquery.Client(project=self.project)
        return self._client

    def _config(self, **kw):
        from google.cloud import bigquery

        return bigquery.QueryJobConfig(**kw)

    def dry_run(self, sql: str) -> dict:
        """Ask BigQuery to plan the query without running it (cache off, so the byte estimate is honest).

        Fails with `bytes_cap` if the estimate exceeds `max_bytes`, so an expensive query is rejected before it costs
        anything. Failures other than syntax and bytes_cap count toward the circuit breaker; successes do not reset it.
        """
        if self.breaker.is_open:
            return {"ok": False, "bytes": 0, "error": "Data source unavailable (circuit open)", "error_kind": "circuit_open"}
        try:
            job = self.client.query(sql, job_config=self._config(dry_run=True, use_query_cache=False))
            job.result(timeout=self.timeout_s)
            total = int(job.total_bytes_processed or 0)
        except Exception as e:  # noqa: BLE001
            kind = classify_error(e)
            if kind not in _NOT_BREAKER_KINDS:
                self.breaker.record_failure()
            return {"ok": False, "bytes": 0, "error": str(e), "error_kind": kind}
        if total > self.max_bytes:
            return {"ok": False, "bytes": total,
                    "error": f"Query would scan {total:,} bytes, above the cap of {self.max_bytes:,}", "error_kind": "bytes_cap"}
        return {"ok": True, "bytes": total, "error": None, "error_kind": None}

    def query(self, sql: str) -> dict:
        """Run the query with `maximum_bytes_billed` set to `max_bytes` and return at most `row_cap` JSON-safe rows,
        with `row_count` and bytes processed. A success closes the circuit breaker's failure count. The cap on rows
        truncates silently; the guard's LIMIT is what normally keeps results under it."""
        if self.breaker.is_open:
            return {"ok": False, "rows": [], "row_count": 0, "bytes": 0,
                    "error": "Data source unavailable (circuit open)", "error_kind": "circuit_open"}
        try:
            job = self.client.query(sql, job_config=self._config(maximum_bytes_billed=self.max_bytes))
            rows = [json_safe(dict(r.items()) if hasattr(r, "items") else r) for r in job.result(timeout=self.timeout_s, max_results=self.row_cap)]
            total = int(getattr(job, "total_bytes_processed", 0) or 0)
        except Exception as e:  # noqa: BLE001
            kind = classify_error(e)
            if kind not in _NOT_BREAKER_KINDS:
                self.breaker.record_failure()
            return {"ok": False, "rows": [], "row_count": 0, "bytes": 0, "error": str(e), "error_kind": kind}
        self.breaker.record_success()
        return {"ok": True, "rows": rows, "row_count": len(rows), "bytes": total, "error": None, "error_kind": None}

    def live_schema(self) -> dict:
        """Fetch `{table: {column: type}}` for the four allowed tables from BigQuery. Raises on failure (callers fall back to the static schema); not behind the circuit breaker."""
        out: dict = {}
        for table in TABLES:
            t = self.client.get_table(f"{self.dataset}.{table}")
            out[table] = {f.name: f.field_type for f in t.schema}
        return out
