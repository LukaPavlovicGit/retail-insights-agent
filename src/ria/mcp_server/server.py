"""FastMCP stdio server exposing BigQuery (read-only) and the saved-reports library.

Run as `python -m ria.mcp_server.server`; the agent starts it as a child process through `ria.tools.open_mcp_toolbox`
and talks to it over stdio. It is the only code that touches BigQuery or the `reports` table at runtime. It
trusts its caller: SQL must already have passed `guards.sql_guard`, because this layer only adds byte caps and
a row cap. Configuration comes from environment variables set by the launcher. Tools return JSON strings.
"""
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
    """Close the report database and drop the cached runner and store, so the next call rebuilds them from the current environment (used by tests)."""
    global _runner, _store
    if _store is not None:
        _store.conn.close()
    _runner, _store = None, None


def runner() -> BigQueryRunner:
    """The process-wide `BigQueryRunner`, created on first use from the RIA_* and GOOGLE_CLOUD_PROJECT env vars.
    Lazy so the server starts, and `ria` works for non-BigQuery tools, before credentials are needed. One runner
    means one circuit breaker shared by all tool calls."""
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
    """The process-wide `ReportStore`, opened lazily on RIA_DB_PATH, the same database file the agent uses (separate connection, WAL mode)."""
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
    """Column names and types for orders, order_items, products, users.

    Returns the live schema (`source: "live"`), or the built-in one with a `warning` (`source: "static"`) if BigQuery cannot be reached."""
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


@mcp.tool()
def reports_get(user_id: str, report_id: int) -> str:
    """One of the user's reports by id, with its question, SQL and body; `{"report": null}` if it does not exist or is not theirs."""
    r = store().get(user_id, report_id)
    if r is None:
        return json.dumps({"report": None})
    return json.dumps({"report": {"id": r.id, "thread_id": r.thread_id, "title": r.title, "question": r.question, "sql": r.sql,
                                  "body": r.body, "tags": r.tags, "created_at": r.created_at}})


def main() -> None:
    """Serve the tools over stdio until the parent process closes the pipe."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
