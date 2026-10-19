"""FastMCP stdio server exposing BigQuery (read-only) and the saved-reports library."""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from ria import db
from ria.guards.schema import STATIC_SCHEMA
from ria.mcp_server.bigquery import BigQueryRunner
from ria.mcp_server.reports import ReportStore

mcp = FastMCP("ria-tools", instructions="Read-only BigQuery access and the saved-reports library for the Retail Insights Agent.", log_level="WARNING")

_runner: BigQueryRunner | None = None
_store: ReportStore | None = None


def reset() -> None:
    global _runner, _store
    if _store is not None:
        _store.conn.close()
    _runner, _store = None, None


def runner() -> BigQueryRunner:
    global _runner
    if _runner is None:
        _runner = BigQueryRunner(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
            dataset=os.environ.get("RIA_DATASET", "bigquery-public-data.thelook_ecommerce"),
            max_bytes=int(os.environ.get("RIA_BQ_MAX_BYTES", "1000000000")),
            timeout_s=int(os.environ.get("RIA_BQ_TIMEOUT_S", "60")),
            row_cap=int(os.environ.get("RIA_ROW_CAP", "200")),
        )
    return _runner


def store() -> ReportStore:
    global _store
    if _store is None:
        conn = db.connect(os.environ.get("RIA_DB_PATH", "ria.db"))
        db.init_schema(conn, int(os.environ.get("RIA_EMBEDDING_DIM", "768")))
        _store = ReportStore(conn)
    return _store


@mcp.tool()
def bq_dry_run(sql: str) -> str:
    """Estimate bytes scanned by a SELECT without running it."""
    return json.dumps(runner().dry_run(sql))


@mcp.tool()
def bq_query(sql: str) -> str:
    """Run a read-only SELECT against the retail dataset; rows are capped."""
    return json.dumps(runner().query(sql))


@mcp.tool()
def bq_schema() -> str:
    """Column names and types for orders, order_items, products, users."""
    try:
        return json.dumps({"source": "live", "schema": runner().live_schema()})
    except Exception as e:  # noqa: BLE001 - no credentials or network: fall back
        return json.dumps({"source": "static", "schema": STATIC_SCHEMA, "warning": str(e)[:200]})


@mcp.tool()
def reports_save(user_id: str, thread_id: str, title: str, question: str, sql: str, body: str, tags: list[str]) -> str:
    """Save a report to the user's library."""
    rid = store().save(user_id, thread_id, title, question, sql, body, tags)
    return json.dumps({"ok": True, "id": rid})


@mcp.tool()
def reports_search(user_id: str, text: str | None = None, thread_id: str | None = None) -> str:
    """List the user's reports, optionally filtered by text or conversation."""
    rows = store().search(user_id, text=text, thread_id=thread_id)
    return json.dumps({"reports": [{"id": r.id, "title": r.title, "created_at": r.created_at, "thread_id": r.thread_id} for r in rows]})


@mcp.tool()
def reports_delete(user_id: str, ids: list[int]) -> str:
    """Delete reports by id; only the user's own reports are affected."""
    return json.dumps({"deleted": store().delete(user_id, ids)})


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
