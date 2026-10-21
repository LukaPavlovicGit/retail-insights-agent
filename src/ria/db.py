"""SQLite schema, connection setup and user records for the whole app.

One database file holds users, saved reports, Golden Trios (plus the `trio_vec` sqlite-vec index), per-user
preferences, turn traces and the LLM ledger. The LangGraph checkpointer in `cli.chat` uses the same file for its own tables.
Both the agent process and the MCP server process open this file (WAL mode), each through `connect`.
Depends only on sqlite-vec and PyYAML; the stores in `golden`, `memory`, `observability` and `mcp_server` build on it.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec
import yaml

DDL = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, scope TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS user_prefs (
  user_id TEXT PRIMARY KEY, format TEXT NOT NULL DEFAULT 'prose', depth TEXT NOT NULL DEFAULT 'standard',
  charts INTEGER NOT NULL DEFAULT 0, notes TEXT NOT NULL DEFAULT '', brevity_streak INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, thread_id TEXT NOT NULL, title TEXT NOT NULL,
  question TEXT NOT NULL, sql TEXT NOT NULL, body TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trios (
  id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, sql TEXT NOT NULL, report TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '["all"]', status TEXT NOT NULL DEFAULT 'promoted', source TEXT NOT NULL DEFAULT 'seed',
  author TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces (
  turn_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_id TEXT NOT NULL, intent TEXT, outcome TEXT NOT NULL,
  started_at TEXT NOT NULL, duration_ms INTEGER NOT NULL, tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0, bq_bytes INTEGER NOT NULL DEFAULT 0, fallback_used INTEGER NOT NULL DEFAULT 0,
  repairs INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT, turn_id TEXT NOT NULL, node TEXT NOT NULL, model TEXT NOT NULL,
  tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL, latency_ms INTEGER NOT NULL,
  fallback INTEGER NOT NULL DEFAULT 0);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the app database with dict-like rows and the sqlite-vec extension loaded.

    `check_same_thread=False` because async code may touch the connection from different threads. WAL is enabled
    for file databases so the agent and the MCP server can share the file; `:memory:` (tests) skips it.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    """Create the tables and the `trio_vec` cosine-distance vector table if missing (idempotent).

    `embedding_dim` is baked into the vector table at creation, so it must match `Settings.embedding_dim`
    and the embedder; changing it later needs a new database.
    """
    conn.executescript(DDL)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS trio_vec USING vec0("
        f"trio_id INTEGER PRIMARY KEY, embedding float[{embedding_dim}] distance_metric=cosine)"
    )
    conn.commit()


def _row_to_user(row: sqlite3.Row) -> dict:
    scope = json.loads(row["scope"])
    return {
        "id": row["id"],
        "name": row["name"],
        "role": row["role"],
        "scope": None if scope == ["*"] else scope,
    }


def load_users(conn: sqlite3.Connection, path: Path) -> list[dict]:
    """Upsert the demo users from users.yaml and return them as user dicts.

    Each entry needs id, name, role and brands, otherwise `ValueError`. `brands: ["*"]` means
    unrestricted and comes back as `None`; any other list is the user's scope (stored in the `scope` column).
    """
    raw = yaml.safe_load(Path(path).read_text()) or []
    for u in raw:
        for key in ("id", "name", "role", "brands"):
            if key not in u:
                raise ValueError(f"{path}: user entry missing '{key}'")
        conn.execute(
            "INSERT OR REPLACE INTO users (id, name, role, scope) VALUES (?, ?, ?, ?)",
            (u["id"], u["name"], u["role"], json.dumps(list(u["brands"]))),
        )
    conn.commit()
    return [get_user(conn, u["id"]) for u in raw]


def get_user(conn: sqlite3.Connection, user_id: str) -> dict | None:
    """Return `{id, name, role, scope}` (scope `None` = unrestricted), or `None` for an unknown id.

    This dict is the `UserCtx` that the graph carries in state and the SQL guard uses for scoping."""
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None
